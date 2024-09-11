import math

import torch
import torch.nn as nn
from transformers import DynamicCache

from ...config import CommonConfig
from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...modeling_utils import get_attention_module, get_normalization_function
from ...modeling_utils.attention import Attention
from ...modeling_utils.linear import ParameterizedLinear
from ..gpt_dolomite.mlp import MLP
from .config import SBDolomiteConfig
from .sb_varlen import sb_attn_varlen_, sb_flash_attn_varlen


class PaddingFreeSBAttention(Attention):
    def __init__(self, config: CommonConfig, causal: bool, layer_idx: int | None = None) -> None:
        super().__init__(config, causal, layer_idx)
        if config.add_qkv_bias:
            init_method = InitMethod(config.init_method)
            initializer_range = config.initializer_range
            m_width = config.m_width
            std = initializer_range
            if init_method == InitMethod.mup:
                std /= math.sqrt(m_width)
            self.c_attn = ParameterizedLinear(
                self.hidden_size,
                self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
                bias=True,
                std=std,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sb_metadata=None,
    ) -> torch.Tensor:
        assert past_key_values is None

        # ==========================================================================================
        # hidden_states -> (total_q, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (total_q, num_heads, head_dim)
        # key -> (total_q, num_key_value_heads, head_dim)
        # value -> (total_q, num_key_value_heads, head_dim)
        # ==========================================================================================

        # if self.position_embedding_type == PositionEmbeddingType.rope:
        #     query = apply_rotary_pos_emb(query, rope_cos_sin)
        #     key = apply_rotary_pos_emb(key, rope_cos_sin)

        # ==========================================================================================
        # query -> (total_q, num_heads, head_dim)
        # key -> (total_q, num_key_value_heads, head_dim)
        # value -> (total_q, num_key_value_heads, head_dim)
        # ==========================================================================================

        softmax_scale = self._get_softmax_scale()
        self.attn_pdrop if self.training else 0
        # attn_output = flash_attn_varlen_func(
        #     query,
        #     key,
        #     value,
        #     cu_seqlens_q=cu_seqlens,
        #     cu_seqlens_k=cu_seqlens,
        #     max_seqlen_q=max_seqlen,
        #     max_seqlen_k=max_seqlen,
        #     dropout_p=dropout_p,
        #     softmax_scale=softmax_scale,
        #     causal=self.causal,
        # )

        # attn_output, rem = sb_flash_attn_varlen(
        #     q=query.permute(1, 0, 2).contiguous(),
        #     k=key.permute(1, 0, 2).contiguous(),
        #     v=v_.contiguous(),
        #     cu_seqlens=torch.tensor([query.size(0)], dtype=torch.int32, device=query.device),
        #     inv_temp=softmax_scale,
        #     zero_start=False,
        # )
        # attn_output = attn_output + rem[..., None] * v_
        # attn_output = attn_output.permute(1, 0, 2).contiguous()
        """
        v_ = value.permute(1, 0, 2)
        cu_row_blocks_, first_row_block_, sequence_ids_ = sb_metadata
        attn_output, rem, sb_metadata_ = sb_flash_attn_varlen(
            q=query.permute(1, 0, 2),
            k=key.permute(1, 0, 2),
            v=v_,
            cu_seqlens=cu_seqlens,
           inv_temp=softmax_scale,
        )
        cu_row_blocks, first_row_block, sequence_ids = sb_metadata_
        assert (cu_row_blocks_ == cu_row_blocks).all(), "cu_row_blocks don't match"
        assert (first_row_block_ == first_row_block).all(), "first_row_block don't match"
        assert (sequence_ids == sequence_ids_).all(), "sequence_idsdon't match"
        """
        cu_row_blocks, first_row_block, sequence_ids = sb_metadata
        v_ = value.permute(1, 0, 2)
        attn_output, rem = sb_attn_varlen_(
            q=query.permute(1, 0, 2),
            k=key.permute(1, 0, 2),
            v=v_,
            inv_temp=softmax_scale,
            cu_seqlens=cu_seqlens[1:],
            first_row_block=first_row_block,
            cu_row_blocks=cu_row_blocks,
            sequence_ids=sequence_ids,
        )
        # attn_output = attn_output + rem[..., None] * v_
        attn_output = attn_output.permute(1, 0, 2)

        # ==========================================================================================
        # attn_output -> (total_q, num_heads, head_dim)
        # ==========================================================================================

        attn_output = attn_output.view(-1, self.hidden_size)

        # ==========================================================================================
        # attn_output -> (total_q, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)
        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(total_q, -1, self.head_dim)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        query = query.view(total_q, self.num_heads, -1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value


class SBDolomiteBlock(nn.Module):
    """
    Layer implementation for the transformer block
    """

    def __init__(
        self,
        config: SBDolomiteConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        assert use_padding_free_transformer

        # self.attn = get_attention_module(
        #     config, True, attention_implementation, use_padding_free_transformer, layer_idx
        # )
        self.attn = PaddingFreeSBAttention(config, causal=True, layer_idx=layer_idx)
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sb_metadata=None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sb_metadata=sb_metadata,
        )

        if self.m_residual is not None:
            attn_output = attn_output * self.m_residual

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)

        if self.m_residual is not None:
            feed_forward_hidden_states = feed_forward_hidden_states * self.m_residual

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states
