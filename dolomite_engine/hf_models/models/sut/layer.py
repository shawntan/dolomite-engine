import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_cute_kernels_available
from ...cache import GenerationCache
from ...mixins.dense.layer import Block
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
from ...modeling_utils.linear import ParameterizedLinear
from ...modeling_utils.mlp_blocks.mlp import MLP, _get_std_for_linear
from ...modeling_utils.mlp_blocks.moe import MoE, compute_bincount
from ...modeling_utils.sequence_mixer_blocks.momha import MoAttention
from ...modeling_utils.sequence_mixer_blocks.softmax_attention import Attention
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from . import mixture_aux_loss
from .config import SUTConfig


def _compute_router_statistics(logits: torch.Tensor, topk_idxs: torch.Tensor, is_hopper_or_newer_gpu) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    logits = logits.view(-1, logits.size(-1))
    probs = probs.view(-1, probs.size(-1))
    num_experts = logits.size(1)

    sum_freq = compute_bincount(
        x=topk_idxs.flatten(),
        size=num_experts,
        use_continuous_count=is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count_cute),
    )
    sum_probs = probs.sum(0)
    sum_lse_sq = (torch.logsumexp(logits, dim=-1) ** 2).sum()
    return sum_freq.to(torch.long), sum_probs, sum_lse_sq


# FIXME refactor to use this.
def _compute_halted_router_statistics(
    logits: torch.Tensor, topk_idxs: torch.Tensor, non_halt_weight: torch.Tensor
) -> torch.Tensor:
    num_experts = logits.size(-1)
    k = topk_idxs.size(-1)
    logits = logits.view(-1, num_experts)
    non_halt_weight = non_halt_weight.flatten().float()
    probs = torch.softmax(logits, dim=-1)
    probs = probs.view(-1, probs.size(-1)).float()
    sum_freq = torch.zeros((num_experts,), dtype=torch.float32, device=logits.device)
    sum_freq.scatter_add_(dim=0, index=topk_idxs.flatten(), src=non_halt_weight.repeat_interleave(k))
    sum_probs = (non_halt_weight.unsqueeze(0).float() @ probs).squeeze(0)
    sum_lse_sq = ((torch.logsumexp(logits, dim=-1) ** 2).unsqueeze(0).float() @ probs).squeeze(0)
    return sum_freq.to(torch.long), sum_probs, sum_lse_sq


class RoutingGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        std: float,
        router_intermediate_size: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.std = std
        self.dropout = nn.Dropout(dropout)

        self.transform = ParameterizedLinear(
            in_features=hidden_size, out_features=num_experts, bias=False, std=self.std
        )
        mark_parameter_as_mup_learning_rate(self.transform.weight)

    def forward(self, x):
        x = self.dropout(x)
        return F.linear(input=x, weight=self.transform.weight)

    def extra_repr(self):
        return f"std={self.std},\n"


class SUTMoAttention(MoAttention):
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
        shared_kv_cache: bool = False,
    ) -> None:
        self.num_experts = num_experts
        self.shared_kv_cache = shared_kv_cache
        super().__init__(
            num_experts,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            attention_multiplier,
            position_embedding_type,
            add_bias,
            softmax_dropout,
            dropout,
            init_method,
            initializer_range,
            m_width,
            num_layers,
            causal,
            layer_idx,
            use_padding_free_transformer,
        )
        if self.num_experts > 1:
            std = _get_std_for_linear(initializer_range, init_method, m_width)
            self.gate = RoutingGate(hidden_size=hidden_size, num_experts=num_experts, std=std)

        if self.shared_kv_cache:
            assert self.num_experts == 1  # TODO complicated to make it work for moa

            # Make sure only compute q
            std = initializer_range
            if init_method == "mup":
                std /= math.sqrt(m_width)
            self.c_attn = ParameterizedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=self.add_bias,
                std=std,
            )
            mark_parameter_as_mup_learning_rate(self.c_attn.weight)

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)
        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)
        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)
        return router_logits, router_weights, selected_experts

    def _prepare_qkv(self, hidden_states, key=None, value=None, kv_hidden_states=None):
        if not self.shared_kv_cache:
            return super()._prepare_qkv(hidden_states, key=key, value=value, kv_hidden_states=kv_hidden_states)
        else:
            if self.use_padding_free_transformer:
                total_q = hidden_states.shape[0]
                # input_shape = (total_q, self.num_key_value_heads, -1)
                output_shape = (total_q, -1, self.head_dim)
            else:
                batch_size, query_length = hidden_states.shape[:-1]
                # input_shape = (batch_size, query_length, self.num_key_value_heads, -1)
                output_shape = (batch_size, query_length, -1, self.head_dim)
            query = self.c_attn(hidden_states)
            query = query.reshape(*output_shape)
            if not self.use_padding_free_transformer:
                query = query.transpose(1, 2)
            return (query, key, value, None, None, None, None, None, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        kv_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, router_logits, selected_experts, key, value = self.compute_attn(
            hidden_states,
            past_key_values,
            attention_mask,
            rope_cos_sin,
            cu_seqlens,
            max_seqlen,
            key,
            value,
            kv_hidden_states=kv_hidden_states,
        )
        if self.num_experts > 1:
            mixture_aux_loss.update_stats(
                self, _compute_router_statistics(router_logits, selected_experts, self.is_hopper_or_newer_gpu)
            )
        return output, key, value


class SUTMoE(MoE):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            shared_intermediate_size,
            num_experts,
            num_experts_per_tok,
            activation_function,
            add_bias,
            dropout,
            init_method,
            initializer_range,
            m_width,
            num_layers,
            use_padding_free_transformer,
        )
        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.gate = RoutingGate(hidden_size=hidden_size, num_experts=num_experts, std=std)

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        moe_output = self._compute_experts(hidden_states, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            hidden_states = moe_output
        else:
            hidden_states = moe_output + self._compute_shared_experts(hidden_states)

        del moe_output

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.view(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        mixture_aux_loss.update_stats(
            self, _compute_router_statistics(router_logits, selected_experts, self.is_hopper_or_newer_gpu)
        )
        return hidden_states


class SUTBlock(Block):
    def __init__(
        self,
        config: SUTConfig,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        num_iters: int = 40,
        shared_kv_cache=False,
    ) -> None:
        # super().__init__(config, use_padding_free_transformer)
        nn.Module.__init__(self)
        self.pre_layernorm = config.pre_layernorm
        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type
        self.shared_kv_cache = shared_kv_cache
        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        # sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type
        seq_block = config.sequence_mixer_blocks[layer_idx]
        sequence_mixer_kwargs = dict(
            hidden_size=config.hidden_size,
            num_attention_heads=seq_block.num_attention_heads,
            num_key_value_heads=seq_block.num_key_value_heads,
            attention_multiplier=seq_block.attention_multiplier,
            position_embedding_type=config.position_embedding_type,
            add_bias=seq_block.add_bias,
            dropout=seq_block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=num_iters,
            causal=True,
            layer_idx=layer_idx,
            softmax_dropout=seq_block.softmax_dropout,
            use_padding_free_transformer=use_padding_free_transformer,
        )

        self.sequence_mixer = SUTMoAttention(
            **sequence_mixer_kwargs,
            num_experts=seq_block.num_experts if seq_block.sequence_mixer_type == "mo_attention" else 1,
            shared_kv_cache=shared_kv_cache,
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        seq_block = config.mlp_blocks[layer_idx]

        mlp_kwargs = dict(
            hidden_size=config.hidden_size,
            intermediate_size=seq_block.intermediate_size,
            activation_function=seq_block.activation_function,
            add_bias=seq_block.add_bias,
            dropout=seq_block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=num_iters,
        )
        if seq_block.mlp_type == "MoE":
            self.mlp_block = SUTMoE(
                **mlp_kwargs,
                shared_intermediate_size=seq_block.shared_intermediate_size,
                num_experts=seq_block.num_experts,
                num_experts_per_tok=seq_block.num_experts_per_tok,
                use_padding_free_transformer=use_padding_free_transformer,
            )
        else:
            self.mlp_block = MLP(**mlp_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        layer_idx: int | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        kv_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.sequence_mixer.layer_idx = layer_idx
        self.mlp_block.layer_idx = layer_idx
        residual = hidden_states

        if self.pre_layernorm:
            hidden_states = self.ln_1(hidden_states)

        hidden_states, key, value = self.sequence_mixer(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            key=key,
            value=value,
            kv_hidden_states=kv_hidden_states,
        )
        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual
        if not self.pre_layernorm:
            hidden_states = self.ln_1(hidden_states)

        residual = hidden_states
        if self.pre_layernorm:
            hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp_block(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        if not self.pre_layernorm:
            hidden_states = self.ln_2(hidden_states)

        return hidden_states, key, value
