import torch
import torch.nn.functional as F
from torch import nn

from ...loss import add_aux_loss
from ...modeling_utils.linear import ParameterizedLinear


class HaltingGate(nn.Module):
    def __init__(
        self, hidden_size: int, std: float, dropout: float = 0.05, aux_coeff: float = 1, ln: nn.Module = None
    ) -> None:
        super().__init__()
        self.std = std
        self.dropout = dropout
        self.aux_coeff = aux_coeff
        self.ln = ln
        self.gate = ParameterizedLinear(in_features=hidden_size, out_features=2, bias=False, std=self.std)
        # add temp?
        self.register_buffer("noisy_bias", torch.tensor([0, 5.0]))
        self.reset_parameters()

    def extra_repr(self):
        return f"aux_coeff={self.aux_coeff},"

    def _gate_update(self, h, prev_log_omg):
        # h = F.dropout(h, p=self.dropout)
        h = self.ln(h)
        z = self.gate(h).float()

        if self.training:
            random_halt = torch.rand_like(z[..., 0]) < self.dropout
            z[random_halt] += self.noisy_bias
            # z[..., 0].masked_fill_(random_halt, 5.)

        log_g_hat = torch.log_softmax(z, dim=-1)
        # print("halt > 0.5", (torch.exp(z[..., 0]) > 0.5).float().mean().item())
        if prev_log_omg is not None:
            cum_log_g = prev_log_omg[..., None] + log_g_hat
        else:
            cum_log_g = log_g_hat
        halted = torch.exp(cum_log_g[..., 0])
        return halted.to(h.dtype), cum_log_g[..., 1]

    def forward(self, prev_h, curr_h, prev_halt_state):
        # print("weight max norm", self.gate.weight.abs().max().item())
        if prev_halt_state is not None:
            prev_halted_h, prev_log_omg = prev_halt_state
        else:
            prev_halted_h, prev_log_omg = 0.0, None

        g, curr_log_omg = self._gate_update(prev_h, prev_log_omg)

        curr_halted_h = prev_halted_h + g[..., None] * prev_h
        # curr_aux_loss = prev_aux_loss + g
        p = torch.exp(curr_log_omg)  # prob of NOT halted =  1 - sum halted probabilities

        # torch.set_printoptions(precision=0.2, linewidth=256)
        # print((p[0] > 0.4).long())
        # print(
        #     f"{(p < 0.25).float().mean().item():.5f} "
        #     f"{((0.25 < p) & (p < 0.50)).float().mean().item():.5f} "
        #     f"{((0.50 < p) & (p < 0.75)).float().mean().item():.5f} "
        #     f"{(0.75 < p).float().mean().item():.5f} "
        #     f"{p.mean().item():.5f}"
        # )

        # FIXME halting loss
        if self.training:
            add_aux_loss(self.aux_coeff * p.mean())

        s = curr_halted_h + p[..., None].to(curr_h.dtype) * curr_h
        return s, p, (curr_halted_h, curr_log_omg)

    def finalise(self, curr_h, prev_halt_state):
        if prev_halt_state is not None:
            prev_halted_h, prev_log_omg = prev_halt_state
        g, _ = self._gate_update(curr_h, prev_log_omg)
        curr_halted_h = prev_halted_h + g[..., None] * curr_h
        return curr_halted_h

    def reset_parameters(self):
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
        # self.gate.bias.data[0] = -5

    # def compute_loss(self, halt_state):
    #     _, log_omg, aux_loss = halt_state
