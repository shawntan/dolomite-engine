import torch
import torch.nn as nn
import torch.nn.functional as F

from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ...gpt_dolomite.mlp import MLP
from ..config import MoEDolomiteConfig


# Copied from transformers.models.jetmoe.modeling_jetmoe.JetMoeTopKGating with JetMoe->GraniteMoe
class GraniteMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        """
        Initialize the top-k gating mechanism.
        Args:
            input_size (`int`):
                Size of the input.
            num_experts (`int`):
                Number of experts.
            top_k (`int`):
                Number of top experts to select.
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, hidden_states):
        # compute the top_k routing decision
        logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
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


class GraniteMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Initialize the GraniteMoeParallelExperts module.
        The experts weights are stored in [num_experts, output_size, input_size] format. Such that it's comptible with
        many MoE libraries, such as [Megablock](https://github.com/databricks/megablocks) and
        [ScatterMoE](https://github.com/shawntan/scattermoe), as well as the
        [MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
        used in vllm.
        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        with torch.no_grad():
            self.weight.normal_()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs, expert_size):
        """
        Forward pass of the GraniteMoeParallelExperts module.
        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.
        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        results = torch.cat(output_list, dim=0)
        return results

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.input_size, self.output_size
        )


class SparseMoE(nn.Module):
    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.normalize_expert_weights = config.normalize_expert_weights
        self.activation_function = config.activation_function
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx
        # self.gate = ParameterizedLinear(self.hidden_size, self.num_experts, bias=False)
        # self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.n_inner

        self.activation = get_activation_function(config.activation_function)
        if config.add_bias:
            self.bias = torch.nn.Parameter(torch.empty(self.hidden_size))

        self.c_fc = GraniteMoeParallelExperts(
            config.num_experts,
            self.hidden_size,
            2 * self.intermediate_size if is_glu(config.activation_function) else self.intermediate_size,
        )
        self.c_proj = GraniteMoeParallelExperts(config.num_experts, self.intermediate_size, self.hidden_size)

        self.gate = GraniteMoeTopKGating(
            input_size=self.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)
        total_q = hidden_states.size(0)

        # router_logits, routing_weights, selected_experts = self._compute_routing_weights(hidden_states)
        # hidden_states = self._compute_expert_outputs(hidden_states, routing_weights, selected_experts)

        _, batch_index, batch_gates, expert_size, router_logits = self.gate(hidden_states)
        expert_inputs = hidden_states[batch_index]

        hidden_states = self.c_fc(expert_inputs, expert_size)
        # chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        # hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        hidden_states = self.activation(hidden_states)
        expert_outputs = self.c_proj(hidden_states, expert_size)

        expert_outputs = expert_outputs * batch_gates.unsqueeze(-1)  # [:, None]
        zeros = torch.zeros((total_q, self.hidden_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        hidden_states = zeros.index_add(0, batch_index, expert_outputs)
        if hasattr(self, "bias"):
            hidden_states = hidden_states + self.bias

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)
            # hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        return hidden_states, router_logits
