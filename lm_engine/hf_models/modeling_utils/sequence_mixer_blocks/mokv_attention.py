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
from ..mlp_blocks.mlp import _get_std_for_linear
from ....utils import ProcessGroupManager, is_xma_available

from ..mlp_blocks.moe import compute_bincount, ParameterizedExperts
from ...loss import add_aux_loss

if is_xma_available():
    from xma import continuous_count

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
        self.num_key_value_heads = num_key_value_heads
        # self.num_key_value_heads = num_attention_heads
        self.add_bias = add_bias
        self.qkv_bias = qkv_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sliding_window = sliding_window
        # self.top_k = num_key_value_heads

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

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.gate = ParameterizedLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads,
            bias=False,
            std=std,
        )


        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_attn_q = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.qkv_bias,
            std=std,
        )
        self.c_attn_kv = ParameterizedExperts(
            num_experts=self.num_heads,
            in_features=self.hidden_size,
            out_features=2 * self.head_dim,
            add_bias=self.qkv_bias,
            std=std,
        )


        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(self.hidden_size, self.hidden_size, bias=self.add_bias, std=std)

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = Dropout(softmax_dropout)
        self.dropout = Dropout(dropout)
        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)



        mark_parameter_as_mup_learning_rate(self.c_attn_q.weight)
        mark_parameter_as_mup_learning_rate(self.c_attn_kv.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, indices = x.topk(self.num_key_value_heads, dim=-1)
        return x, indices
    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        # if self.normalized_topk:
        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)
        router_weights = router_weights.type_as(hidden_states)
        # else:
        #     router_weights = F.softmax(router_logits.float(), dim=-1)
        #     router_weights = router_weights.type_as(hidden_states)
        #     router_weights, selected_experts = self._get_topk(router_weights)

        return router_logits, router_weights, selected_experts
    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        expert_frequency = expert_frequency.float()

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            expert_frequency = all_reduce(
                expert_frequency, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group()
            )

        switch_loss = (
            num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()
        )
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss

        return loss.type_as(logits)





    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        # use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
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

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
            expert_frequency = compute_bincount(
                x=sorted_expert_idxs,
                size=self.num_heads,
                use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count),
            )
            expert_offsets = expert_frequency.cumsum(-1)

        q_heads = self.c_attn_q(hidden_states)
        
        kv_heads = self.c_attn_kv(
            input=hidden_states.flatten(0, 1),
            num_experts_per_token=self.num_key_value_heads,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            grouped_out=False,
        )

        query = q_heads.view(*output_shape)
        # print("Before reshape:", kv_heads.size())
        kv_heads = kv_heads.view(*input_shape)
        # print("After reshape: ", kv_heads.size())
        key, value = kv_heads.chunk(2, dim=-1)
        value = router_weights.unsqueeze(-1) * value
        # query, key, value = (
        #     contiguous_split if Accelerator.get_accelerator() == Accelerator.trainium else torch.split
        # )(
        #     hidden_states,
        #     ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim),
        #     dim=-1,
        # )

        # print(query.size())

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
        # kv_idxs = torch.arange(self.num_heads, dtype=torch.int32, device=query.device)[None, None, :].repeat(batch_size, query_length, 1)
        # print(kv_idxs.size(), selected_experts.size())
        hidden_states = mokv_attention(
            q=query, # .contiguous(),
            k=key, # .contiguous(),
            v=value, # .contiguous(),
            kv_idxs=selected_experts,
            causal=self.causal,
            sm_scale=self.attention_multiplier
        )

        del query, key, value

        hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
        hidden_states = hidden_states.transpose(1, 2).view(*output_shape)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits,
                probs=torch.softmax(router_logits, dim=-1),
                expert_frequency=expert_frequency
            )
            if self.training
            else 0
        )

 
        add_aux_loss(aux_loss)


        return hidden_states
