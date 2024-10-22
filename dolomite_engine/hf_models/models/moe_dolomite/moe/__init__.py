

from dolomite_engine.hf_models.enums import InitMethod
from dolomite_engine.hf_models.modeling_utils import ParameterizedTransposedLinear
from dolomite_engine.hf_models.models.moe_dolomite.config import MoEDolomiteConfig
from dolomite_engine.hf_models.models.moe_dolomite.moe.utils import get_moe


import torch
import torch.nn as nn
import torch.nn.functional as F


import math

from ..config import MoEDolomiteConfig
from .base import Experts
from .scatter import ScatterExperts




class MoEMLP(nn.Module):
    def __init__(
        self, config: MoEDolomiteConfig, moe_implementation: str,
        use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        super().__init__()

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size

        initializer_range = config.initializer_range
        m_width = config.m_width
        init_method = InitMethod(config.init_method)
        residual_dropout = config.resid_pdrop

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.gate = ParameterizedTransposedLinear(
            in_features=self.hidden_size,
            out_features=config.num_experts,
            bias=False,
            std=std,
        )

        self.experts = get_moe(
            config,
            moe_implementation=moe_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )


        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        hidden_states = self.experts(hidden_states, router_weights, selected_experts)

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = self._compute_switch_loss(
            logits=router_logits, probs=torch.softmax(router_logits, dim=-1), topk_idxs=selected_experts
        )

        return hidden_states, router_logits, aux_loss

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices

    def _compute_switch_loss(self, logits: torch.Tensor, probs: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)
        freq = torch.bincount(topk_idxs.flatten(), minlength=num_experts).to(dtype=logits.dtype)

        switch_loss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss

        return loss