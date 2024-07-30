import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ..config import MoEDolomiteConfig


class TopKGating(ParameterizedLinear):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, std: float | None = None) -> None:
        self.num_experts = num_experts
        self.top_k = top_k

        super().__init__(hidden_size, num_experts, bias=False, std=std)

    def forward(self, hidden_states):
        # compute the top_k routing decision
        logits = super().forward(hidden_states).float()  # [batch_size x seq_len, num_experts]
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

        # compute number of input given to each expert
        zeros = torch.zeros(
            [top_k_gates.size(0), self.num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
        )  # [num_tokens, num_experts]
        gates = zeros.scatter(1, top_k_indices, 1)  # [num_tokens, num_experts]
        expert_size = gates.long().sum(0)  # [num_experts,]
        expert_size = expert_size.tolist()

        # sort and group input tokens according to expert assignment
        top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
        batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

        return index_sorted_experts, batch_index, batch_gates, expert_size, logits


class ParameterizedExperts(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int, std: float) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(self, input: torch.Tensor, expert_size: int) -> torch.Tensor:
        input = input.split(expert_size, dim=0)
        input = [F.linear(input[i], self.weight[i]) for i in range(self.num_experts)]
        input = torch.cat(input, dim=0)
        return input

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)


class SparseMoE(nn.Module):
    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        super().__init__()

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.normalize_expert_weights = config.normalize_expert_weights
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.n_inner

        activation_function = config.activation_function

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
        init_method = InitMethod(config.init_method)

        self.gate = TopKGating(
            hidden_size=self.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            std=config.initializer_range,
        )

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = ParameterizedExperts(
            num_experts=config.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedExperts(
            num_experts=config.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            std=std,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)
        total_q = hidden_states.size(0)

        _, batch_index, batch_gates, expert_size, router_logits = self.gate(hidden_states)
        expert_inputs = hidden_states[batch_index]

        hidden_states = self.c_fc(expert_inputs, expert_size)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states, expert_size)

        hidden_states = hidden_states * batch_gates.unsqueeze(-1)  # [:, None]
        zeros = torch.zeros((total_q, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = zeros.index_add(0, batch_index, hidden_states)

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        return hidden_states, router_logits
