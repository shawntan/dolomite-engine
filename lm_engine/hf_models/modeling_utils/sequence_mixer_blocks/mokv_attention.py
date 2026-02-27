# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import Accelerator, divide_if_divisible, is_torch_xla_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate
from ..chunk import contiguous_split
from ..dropout import Dropout
from ..linear import ParameterizedLinear
from ..position_embedding import apply_rotary_pos_emb
from .utils import flash_attention

from .mokv.triton import attention as mokv_attention

class MoKVAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        sliding_window: int | None,
        position_embedding_type: str,
        add_bias: bool,
        qkv_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> MoKVAttention:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        # self.num_key_value_heads = num_key_value_heads
        self.num_key_value_heads = num_attention_heads
        self.add_bias = add_bias
        self.qkv_bias = qkv_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sliding_window = sliding_window

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        divide_if_divisible(
            self.num_heads,
            self.num_key_value_heads,
            f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` ({self.num_key_value_heads})",
        )

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_attn = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            bias=self.qkv_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(self.hidden_size, self.hidden_size, bias=self.add_bias, std=std)

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = Dropout(softmax_dropout)
        self.dropout = Dropout(dropout)

        mark_parameter_as_mup_learning_rate(self.c_attn.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
        accelerator = Accelerator.get_accelerator()
        # print("HEllo using new mokv_attention")
        # if self.use_padding_free_transformer:
        #     assert use_flash_attention_2 or use_flash_attention_3
        #     assert past_key_values is None

        #     total_q = hidden_states.shape[0]
        #     input_shape = (total_q, self.num_key_value_heads, -1)
        #     output_shape = (total_q, -1, self.head_dim)
        # else:
        assert not self.use_padding_free_transformer
        batch_size, query_length = hidden_states.shape[:-1]

        input_shape = (batch_size, query_length, self.num_key_value_heads, -1)
        output_shape = (batch_size, query_length, -1, self.head_dim)

        hidden_states = self.c_attn(hidden_states)
        hidden_states = hidden_states.view(*input_shape)

        query, key, value = (
            contiguous_split if Accelerator.get_accelerator() == Accelerator.trainium else torch.split
        )(
            hidden_states,
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim),
            dim=-1,
        )

        query = query.reshape(*output_shape)

        if not self.use_padding_free_transformer:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=self.layer_idx)

        assert accelerator == Accelerator.cuda

        # if self.use_padding_free_transformer:
        #     output_shape = (-1, self.hidden_size)
        # else:
        #     query = query.transpose(1, 2)
        #     key = key.transpose(1, 2)
        #     value = value.transpose(1, 2)

        output_shape = (batch_size, query_length, -1)

        query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
        key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
        value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

        # hidden_states = flash_attention(
        #     query=query,
        #     key=key,
        #     value=value,
        #     cu_seqlens=cu_seqlens,
        #     max_seqlen=max_seqlen,
        #     attention_mask=attention_mask,
        #     use_padding_free_transformer=self.use_padding_free_transformer,
        #     causal=self.causal,
        #     dropout=self.softmax_dropout_p if self.training else 0,
        #     softmax_scale=self.attention_multiplier,
        #     sliding_window=self.sliding_window,
        # )
        assert self.causal
        kv_idxs = torch.arange(self.num_heads, dtype=torch.int32, device=query.device)[None, None, :].repeat(batch_size, query_length, 1)
        hidden_states = mokv_attention(
            q=query, # .contiguous(),
            k=key, # .contiguous(),
            v=value, # .contiguous(),
            kv_idxs=kv_idxs,
            causal=self.causal,
            sm_scale=self.attention_multiplier
        )

        del query, key, value

        hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
        hidden_states = hidden_states.transpose(1, 2).view(*output_shape)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
