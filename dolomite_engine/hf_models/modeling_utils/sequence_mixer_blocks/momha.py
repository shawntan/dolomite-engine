import math

import torch
from torch import nn
from torch.distributed._functional_collectives import all_reduce
from torch.nn import functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import ProcessGroupManager, divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ...loss import add_aux_loss
from ..linear import ParameterizedLinear
from ..mlp_blocks.mlp import _get_std_for_linear
from ..mlp_blocks.moe import ParameterizedExperts, compute_bincount
from ..position_embedding import apply_rotary_pos_emb
from .flash_attention_utils import flash_attention
from .softmax_attention import Attention


class MoAttention(Attention):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        position_embedding_type: str,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        nn.Module.__init__(self)

        self.causal = causal
        self.hidden_size = hidden_size

        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.add_bias = add_bias

        assert use_padding_free_transformer
        self.use_padding_free_transformer = use_padding_free_transformer

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        assert self.num_heads % self.num_key_value_heads == 0
        self.num_experts = num_experts
        self.top_k = self.num_heads // self.num_key_value_heads

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.gate = ParameterizedLinear(
            in_features=self.hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        self.attention_head_type = "mha"

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        self.q_proj = ParameterizedExperts(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            add_bias=add_bias,
            std=std,
        )

        self.kv_proj = ParameterizedLinear(
            self.hidden_size,
            2 * self.num_key_value_heads * self.head_dim,
            bias=self.add_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)

        self.o_proj = ParameterizedExperts(
            num_experts=num_experts,
            in_features=self.num_key_value_heads * self.head_dim,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

        self.softmax_dropout_p = softmax_dropout
        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

        # torch.nn.init.zeros_(self.q_proj.weight)
        # idxs = 10 * torch.arange(num_experts)[:, None]  + torch.arange(self.num_key_value_heads)[None, :]
        # self.q_proj.weight.data[:, ::self.head_dim, 0] = idxs

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)
        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)
        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)
        return router_logits, router_weights, selected_experts

    def _prepare_qkv_for_forward(self, hidden_states, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets):
        kv_states = self.kv_proj(hidden_states)
        if self.use_padding_free_transformer:
            total_q = hidden_states.shape[0]
            input_shape = (hidden_states.shape[0], self.num_key_value_heads, -1)
            output_shape = (total_q, -1, self.head_dim)
        else:
            batch_size, query_length = hidden_states.shape[:-1]
            input_shape = (*hidden_states.shape[:-1], self.num_key_value_heads, -1)
            output_shape = (batch_size, query_length, -1, self.head_dim)

        kv_states = kv_states.view(*input_shape)
        # query, key, value = hidden_states.split(
        #     ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        # )
        key, value = kv_states.chunk(2, dim=-1)
        # head_idx = torch.arange(self.num_key_value_heads, device=value.device)
        # value = value * 0  + head_idx[None, :, None] # TODO

        # key = torch.repeat_interleave(key, dim=1, repeats=self.num_heads // self.num_key_value_heads)
        # value = torch.repeat_interleave(value, dim=1, repeats=self.num_heads // self.num_key_value_heads)

        query = self.q_proj(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_out=False,
        )
        # group heads for gqa-style attention computation
        query = query.view(total_q, self.top_k, self.num_key_value_heads, self.head_dim).transpose(2, 1)
        query = query.reshape(*output_shape)

        if not self.use_padding_free_transformer:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        return query, key, value


    def extra_repr(self) -> str:
        return f"num_attention_heads={self.num_attention_heads},\nnum_key_value_heads={self.num_key_value_heads}"


    def compute_attn(
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

        if self.use_padding_free_transformer:
            assert use_flash_attention_2 or use_flash_attention_3
            assert past_key_values is None

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
            expert_offsets = compute_bincount(
                x=sorted_expert_idxs,
                size=self.num_experts,
                use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count_cute),
            ).cumsum(-1)

        query, key, value = self._prepare_qkv_for_forward(
            hidden_states, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets
        )

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=self.layer_idx)

        if self.use_padding_free_transformer:
            (-1, self.hidden_size)
        else:
            # TODO avoid this extra transpose
            query = query.transpose(1, 2)
            if self.attention_head_type == "mqa":
                key = key.squeeze(1).unsqueeze(2)
                value = value.squeeze(1).unsqueeze(2)
            else:
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            batch_size, query_length = query.shape[:2]
            (batch_size, query_length, -1)

        # query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
        # key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
        # value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)
        hidden_states = flash_attention(
            query=query,
            key=key,
            value=value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            use_padding_free_transformer=self.use_padding_free_transformer,
            causal=self.causal,
            dropout=self.softmax_dropout_p if self.training else 0,
            softmax_scale=self._get_softmax_scale(),
        )
        # hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
        hidden_states = hidden_states.view(-1, self.num_key_value_heads, self.top_k, self.head_dim).transpose(2, 1)
        hidden_states = hidden_states.reshape(
            hidden_states.size(0) * self.top_k, self.num_key_value_heads * self.head_dim
        )
        hidden_states = self.o_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            gates=router_weights,
        )
        hidden_states = self.dropout(hidden_states)
        return hidden_states, router_logits, selected_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        output, router_logits, selected_experts = self.compute_attn(
            hidden_states,
            past_key_values,
            attention_mask,
            rope_cos_sin,
            cu_seqlens,
            max_seqlen,
        )
        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits, probs=torch.softmax(router_logits, dim=-1), topk_idxs=selected_experts
            )
            if self.training
            else 0
        )
        add_aux_loss(aux_loss)
        return output

    def _compute_switch_loss(self, logits: torch.Tensor, probs: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        freq = compute_bincount(
            x=topk_idxs.flatten(),
            size=num_experts,
            use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count_cute),
        )

        freq = freq.float()

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            freq = all_reduce(freq, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group())

        switch_loss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss

        return loss.type_as(logits)
