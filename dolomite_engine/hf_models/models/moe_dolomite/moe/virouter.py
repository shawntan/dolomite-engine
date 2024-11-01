import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .....utils import is_kernel_hyperdrive_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedTransposedLinear, get_activation_function, is_glu
from ..config import MoEDolomiteConfig
from .scatter import ScatterMoE


def gaussian_kl(mean_1, log_std_1, mean_2, log_std_2):
    log_std_diff = log_std_2 - log_std_1
    return torch.sum(
        log_std_diff
        + 0.5 * (torch.exp(-2 * log_std_diff) + ((mean_1 - mean_2) ** 2) / torch.exp(2 * log_std_2))
        - 0.5,
        dim=-1,
    )


class VarMoE(ScatterMoE):

    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        super().__init__(config, use_padding_free_transformer, layer_idx)
        initializer_range = config.initializer_range
        m_width = config.m_width

        self.router_aux_loss_coeff = config.router_aux_loss_coef

        self.kl_coeff = 1.0 / self.router_aux_loss_coeff
        self.register_buffer("log_logit_std", torch.log(torch.tensor(0.5)))

        init_method = InitMethod(config.init_method)
        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        self.q_gate = ParameterizedTransposedLinear(
            in_features=self.hidden_size,
            out_features=config.num_experts,
            bias=False,
            std=std,
        )

        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.q_gate.weight)

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[Tensor]:
        # hidden_states -> (total_q, hidden_size)

        if self.training:
            assert self.use_padding_free_transformer
            p_mean = self.gate(hidden_states)
            q_mean_ = torch.cat((self.q_gate(hidden_states[1:]), self.gate(hidden_states[-1:])), dim=0).detach()
            q_mean = 0.5 * (q_mean_ + p_mean)
            q_logits = q_mean + torch.exp(self.log_logit_std) * torch.randn_like(q_mean)
        else:
            q_mean = self.gate(hidden_states)
            q_logits = q_mean_ = p_mean = q_mean

        router_weights, selected_experts = self._get_topk(q_logits)
        # router_logits -> (total_q, num_experts)
        router_weights = F.softmax(router_weights.float(), dim=-1)
        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)
        return q_logits, p_mean, q_mean, q_mean_, router_weights, selected_experts

    def _compute_gate_kl_divergence(self, p_mean, q_mean):
        """
        p_gate_logits = p_gate_logits.view(-1, self.num_experts)
        q_gate_logits = q_gate_logits.view(-1, self.num_experts)
        log_p = F.log_softmax(p_gate_logits, dim=-1)
        log_q = F.log_softmax(q_gate_logits, dim=-1)
        kl = torch.bmm(
            torch.exp(log_q)[:, None, :],
            (log_q - log_p)[:, :, None]
        )[:, 0, 0]
        """
        kl = gaussian_kl(mean_1=q_mean, log_std_1=self.log_logit_std, mean_2=p_mean, log_std_2=self.log_logit_std)
        kl_loss = self.kl_coeff * torch.mean(kl, dim=0)
        return kl_loss

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        q_logits, p_mean, q_mean, q_mean_, router_weights, selected_experts = self._compute_routing_weights(
            hidden_states
        )
        hidden_states = self._compute_experts(hidden_states, router_weights, selected_experts)

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = self._compute_switch_loss(
            logits=q_mean_,  # only q_router_logits need to be z-lossed
            probs=torch.softmax(q_logits, dim=-1),
            topk_idxs=selected_experts,  # regularise the actual q to maintain spread
        )
        aux_loss = aux_loss + self._compute_gate_kl_divergence(p_mean=p_mean, q_mean=q_mean)
        return hidden_states, q_logits, aux_loss
