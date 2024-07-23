import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....utils import is_scattermoe_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ..config import MoEDolomiteConfig


if is_scattermoe_available():
    import scattermoe
    from scattermoe.parallel_experts import ParallelExperts


class ScatterMoE(nn.Module):
    def __init__(self, config: MoEDolomiteConfig, use_padding_free_transformer: bool) -> None:
        super().__init__()

        hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.normalize_expert_weights = config.normalize_expert_weights

        # router
        self.gate = ParameterizedLinear(hidden_size, self.num_experts, bias=False)
        self.experts = _ScatterMoEMLP(config)

        self.use_padding_free_transformer = use_padding_free_transformer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_padding_free_transformer:
            _, hidden_dim = hidden_states.shape
        else:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits -> (batch_size * sequence_length, num_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        if self.top_k == 1:
            routing_weights, selected_experts = routing_weights.max(dim=-1, keepdim=True)
        else:
            routing_weights, selected_experts = routing_weights.topk(self.top_k, dim=-1)

        if self.normalize_expert_weights:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = self.experts(hidden_states, routing_weights, selected_experts)

        if not self.use_padding_free_transformer:
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits


class _ScatterMoEMLP(nn.Module):
    def __init__(self, config: MoEDolomiteConfig) -> None:
        super().__init__()

        hidden_size = config.n_embd
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        residual_dropout = config.resid_pdrop

        assert not config.add_bias, "ScatterMoE does not support add_bias"

        self.init_method = InitMethod(config.init_method)
        self.initializer_range = config.initializer_range
        self.m_width = config.m_width
        self.n_layer = config.n_layer

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.c_fc = ParallelExperts(
            self.num_experts, hidden_size, 2 * intermediate_size if is_glu(activation_function) else intermediate_size
        )
        # ParameterizedLinear(
        #     hidden_size,
        #     2 * intermediate_size if is_glu(activation_function) else intermediate_size,
        #     bias=add_bias,
        #     std=std,
        # )

        self.act = get_activation_function(activation_function)

        self.c_proj = ParallelExperts(self.num_experts, intermediate_size, hidden_size)
        # ParameterizedLinear(intermediate_size, hidden_size, bias=add_bias, std=std)
        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

        self.reset_parameters()

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True,
        )

        hidden_states = self.act(hidden_states)

        hidden_states = self.c_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_in=True,
            gates=routing_weights,
        )

        hidden_states = self.dropout(hidden_states)

        return hidden_states

    @torch.no_grad()
    def reset_parameters(self) -> None:
        std = self.initializer_range
        if self.init_method == InitMethod.mup:
            std /= math.sqrt(self.m_width)
        nn.init.normal_(self.c_fc.weight, mean=0, std=std)

        std = self.initializer_range / math.sqrt(2 * self.n_layer)
        if self.init_method == InitMethod.mup:
            std /= math.sqrt(self.m_width)
        nn.init.normal_(self.c_proj.weight, mean=0, std=std)
