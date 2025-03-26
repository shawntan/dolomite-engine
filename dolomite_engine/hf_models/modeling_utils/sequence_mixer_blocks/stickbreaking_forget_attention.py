import math

import torch
import torch.nn
import torch.nn.functional as F
from transformers import DynamicCache
import logging
from ....utils import is_stickbreaking_available, log_rank_0
from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from .softmax_attention import Attention

from ..linear import ParameterizedLinear


if is_stickbreaking_available():
    from stickbreaking_attention import sb_forget_varlen, sb_varlen

try:
    import os
    FORGET_THRESHOLD = float(os.environ.get("FORGET_THRESHOLD"))
    log_rank_0(logging.INFO, f"set FORGET_THRESHOLD={FORGET_THRESHOLD}")
except:
    FORGET_THRESHOLD = None


def decoding_stickbreaking(q, k, v, scale=None):
    """
    Stick-breaking attention weights.
    """
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])
    # logits = q @ k[..., :-1, :].transpose(-1, -2) * scale

    assert q.size(2) == 1
    original_dtype = q.dtype
    q = q.float()
    k = k.float()
    # logits = torch.einsum('bhid,bhjd->bhij', q, k[..., :-1, :]) * scale
    logits = q @ k[..., :-1, :].transpose(-1, -2) * scale
    # logits = logits.float()
    log_z = F.logsigmoid(logits).to(original_dtype)
    log_beta = F.logsigmoid(-logits).to(original_dtype)
    # re_cum_log_beta = log_beta.sum(dim=-1, keepdim=True) - log_beta.cumsum(dim=-1)
    re_cum_log_beta = log_beta.flip(-1).cumsum(dim=-1).flip(-1) - log_beta
    # re_cum_log_beta = log_beta.sum(dim=-1, keepdim=True) - log_beta.cumsum(dim=-1)
    log_att = log_z + re_cum_log_beta
    # print("log_att", log_att[0, 0, 0, -20:])
    att = log_att.exp()
    #  print("att    ", att[0, 0, 0, -20:])
    out = torch.einsum("bhij,bhjd->bhid", att, v[..., :-1, :])
    # out = att @ v[..., :-1, :]
    return out, 1 - att.sum(dim=-1)

class SBForgetAttention(Attention):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        add_bias: bool,
        dropout: float,
        init_method: InitMethod,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        head_bias: bool,
        out_norm: bool ,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_multiplier=attention_multiplier,
            attention_head_type=attention_head_type,
            position_embedding_type=position_embedding_type,
            add_bias=add_bias,
            softmax_dropout=0,
            dropout=dropout,
            init_method=init_method,
            initializer_range=initializer_range,
            m_width=m_width,
            num_layers=num_layers,
            causal=causal,
            layer_idx=layer_idx,
        )
        self.forget_gate = ParameterizedLinear(self.hidden_size, 1, bias=False)
        torch.nn.init.zeros_(self.forget_gate.weight)
        print("head_bias", head_bias, "out_norm", out_norm)
        if head_bias:
            self.head_bias = torch.nn.Parameter(torch.zeros(self.hidden_size // self.head_dim, self.head_dim))
        else:
            self.head_bias = None

        if out_norm:
            self.norm = torch.nn.GroupNorm(self.num_heads, self.hidden_size)
        else:
            self.norm = None

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
        global FORGET_THRESHOLD
        # assert past_key_values is None
        query, key, value = self._prepare_qkv_for_forward(hidden_states)
        softmax_scale = self._get_softmax_scale()

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        bsz_, _, length_, _ = query.size()
        log_forget = F.logsigmoid(self.forget_gate(hidden_states))
        if FORGET_THRESHOLD is not None:
            f = torch.exp(log_forget)
            mask = f < FORGET_THRESHOLD
            log_forget = log_forget.masked_fill(mask, -1000.0)
        log_forget = log_forget.expand(-1, -1, query.size(1)).permute(0, 2, 1).contiguous()

        if query.size(2) == key.size(2):
            def attn_1(query, key, value, softmax_scale):
                bsz, num_heads, length, h_dim = query.size()
                query_ = query.flatten(0, 1)
                key_ = key.flatten(0, 1)
                value_ = value.flatten(0, 1)
                log_forget_ = log_forget.flatten(0, 1)
                hidden_states_, rem_ = sb_forget_varlen.sb_attn_varlen(
                    q=query_,
                    k=key_,
                    v=value_,
                    log_forget=log_forget_,
                    inv_temp=softmax_scale,
                    cu_seqlens=torch.tensor([0, length], device=torch.cuda.current_device()),
                    max_seqlens=length,
                )
                # assert hidden_states_.size(0) == bsz * num_heads, (hidden_states_.size(0), bsz, num_heads)
                hidden_states = hidden_states_.view(bsz, num_heads, length, h_dim)
                rem = rem_.view(bsz, num_heads, length)
                return hidden_states, rem
            hidden_states, rem = attn_1(query, key, value, softmax_scale)
        else:
            raise NotImplementedError("Cannot decode")
            hidden_states, rem = decoding_stickbreaking(q=query, k=key, v=value, scale=softmax_scale)

        if self.head_bias:
            hidden_states = hidden_states + rem[..., None] * self.head_bias[None, :, None, :]

        hidden_states = hidden_states.permute(0, 2, 1, 3)

        if self.norm is not None:
            hidden_states = hidden_states.reshape(bsz_ * length_, self.hidden_size)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.view(bsz_, length_, self.hidden_size)
        else:
            hidden_states = hidden_states.reshape(bsz_, length_, self.hidden_size)


        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(batch_size, query_length, -1, self.head_dim)

        group_size = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats=group_size, dim=2)
        value = value.repeat_interleave(repeats=group_size, dim=2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(self, hidden_states):
        raise NotImplementedError()


class PaddingFreeSBForgetAttention(SBForgetAttention):
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
        query, key, value = self._prepare_qkv_for_forward(hidden_states)
        value = value.permute(1, 0, 2)
        log_forget = F.logsigmoid(self.forget_gate(hidden_states))
        log_forget = log_forget.expand(-1, query.size(1)).permute(1, 0)
        hidden_states, rem = sb_forget_varlen.sb_attn_varlen(
            q=query.permute(1, 0, 2),
            k=key.permute(1, 0, 2),
            v=value,
            log_forget=log_forget,
            inv_temp=self._get_softmax_scale(),
            cu_seqlens=cu_seqlens,
            max_seqlens=max_seqlen,
        )
        if self.head_bias is not None:
            hidden_states = hidden_states + rem[..., None] * self.head_bias[:, None, :]

        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

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
        # key = key.repeat(1, self.num_heads // self.num_key_value_heads, 1)
        # value = value.repeat(1, self.num_heads // self.num_key_value_heads, 1)
        group_size = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats=group_size, dim=1)
        value = value.repeat_interleave(repeats=group_size, dim=1)
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
