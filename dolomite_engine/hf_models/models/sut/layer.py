import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
from transformers import DynamicCache

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import ProcessGroupManager, is_cute_kernels_available
from ...cache import GenerationCache
from ...loss import add_aux_loss
from ...mixins.dense.layer import Block
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
from ...modeling_utils.mlp_blocks.moe import MoE, compute_bincount

from ...modeling_utils.linear import ParameterizedLinear
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear

from .config import SUTConfig

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
        out_linear = ParameterizedLinear(
            in_features=128,
            out_features=num_experts,
            bias=False,
            std=std,
        )
        self.gate = nn.Sequential(
            ParameterizedLinear(
                in_features=self.hidden_size,
                out_features=128,
                bias=False,
                std=std,
            ),
            nn.Tanh(),
            nn.Dropout(0.2),
            out_linear
        )

    def __get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_x = x
        with torch.no_grad():
            if self.training:
                x = x + 0.05 * torch.randn_like(x)
            _, indices = x.topk(self.top_k, dim=-1)
        idxs = torch.arange(x.size(0), dtype=torch.long, device=orig_x.device)
        x = orig_x[idxs[:, None], indices]
        return x, indices

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = 0.025 * self.gate(hidden_states)
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

        # del moe_output

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        return hidden_states, self._compute_router_statistics(router_logits, selected_experts)

    def _compute_router_statistics(self, logits: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))
        num_experts = logits.size(1)

        sum_freq = compute_bincount(
            x=topk_idxs.flatten(),
            size=num_experts,
            use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count_cute),
        )
        sum_probs = probs.sum(0)
        sum_lse_sq = (torch.logsumexp(logits, dim=-1) ** 2).sum()
        return sum_freq.to(torch.long), sum_probs, sum_lse_sq

    def _update_statistics(self, acc_stats, stats):
        if acc_stats is None:
            return stats
        else:
            acc_freq, acc_probs, acc_lse_sq = acc_stats
            sum_freq, sum_probs, sum_lse_sq = stats
            acc_freq = acc_freq + sum_freq
            acc_probs = acc_probs + sum_probs
            acc_lse_sq = acc_lse_sq + sum_lse_sq
            return acc_freq, acc_probs, acc_lse_sq

    def _compute_switch_loss(self, acc_stats):
        acc_freq, acc_probs, acc_lse_sq = acc_stats
        num_experts = acc_freq.size(0)
        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            acc_freq = all_reduce(acc_freq, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group())
            acc_probs = all_reduce(acc_probs, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group())
        switch_loss = (
            num_experts * (
                F.normalize(acc_probs, p=1, dim=0) *
                F.normalize(acc_freq.float(), p=1, dim=0)
            ).sum()
        )
        z_loss = acc_lse_sq / acc_freq.sum()
        loss = switch_loss + 0.1 * z_loss
        return loss.type_as(acc_lse_sq)


class SUTBlock(Block):
    def __init__(self, config: SUTConfig, use_padding_free_transformer: bool, layer_idx: int | None = None) -> None:
        # super().__init__(config, use_padding_free_transformer)
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.sequence_mixer = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        block = config.mlp_blocks[layer_idx]

        kwargs = dict(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
        )

        self.mlp_block = SUTMoE(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        self.sequence_mixer.layer_idx = layer_idx
        self.mlp_block.layer_idx = layer_idx
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self._sequence_mixer_forward(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states, mlp_router_statistics = self.mlp_block(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states, mlp_router_statistics

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     past_key_values: GenerationCache | None = None,
    #     attention_mask: torch.Tensor | None = None,
    #     rope_cos_sin: torch.Tensor | None = None,
    #     cu_seqlens: torch.Tensor | None = None,
    #     max_seqlen: int | None = None, layer_idx: int | None = None,
    # ) -> torch.Tensor:
    #     self.sequence_mixer.layer_idx = layer_idx
    #     self.mlp_block.layer_idx = layer_idx

    #     x = self.ln_1(hidden_states)
    #     out = self._sequence_mixer_forward(
    #         hidden_states=x,
    #         past_key_values=past_key_values,
    #         attention_mask=attention_mask,
    #         rope_cos_sin=rope_cos_sin,
    #         cu_seqlens=cu_seqlens,
    #         max_seqlen=max_seqlen,
    #     )
    #     hidden_states = self.ln_2(out + x)
    #     x = hidden_states
    #     out, mlp_router_statistics = self.mlp_block(x)
    #     hidden_states = out + x

    #     return hidden_states, mlp_router_statistics
