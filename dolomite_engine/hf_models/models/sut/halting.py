import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_utils.linear import ParameterizedLinear


class HaltingGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        std: float,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.std = std
        self.dropout = dropout
        self.gate = ParameterizedLinear(in_features=hidden_size, out_features=2, bias=True, std=self.std)
        nn.init.zeros_(self.gate.weight)

    def _gate_update(self, h, prev_log_omg):
        # h = F.dropout(h, p=self.dropout)
        z = self.gate(0.0 * h).float()
        z[..., 0] -= 5.0
        log_g_hat = torch.log_softmax(z, dim=-1)
        if prev_log_omg is not None:
            cum_log_g = prev_log_omg[..., None] + log_g_hat
        else:
            cum_log_g = log_g_hat
        halted = torch.exp(cum_log_g[..., 0]).to(h.dtype)
        return halted, cum_log_g[..., 1]

    def forward(self, prev_h, curr_h, prev_halt_state):
        if prev_halt_state is not None:
            prev_halted_h, prev_log_omg = prev_halt_state
        else:
            prev_halted_h, prev_log_omg = 0.0, None

        g, curr_log_omg = self._gate_update(prev_h, prev_log_omg)

        curr_halted_h = prev_halted_h + g[..., None] * prev_h
        s = curr_halted_h + torch.exp(curr_log_omg)[..., None] * curr_h

        return s, (curr_halted_h, curr_log_omg)
