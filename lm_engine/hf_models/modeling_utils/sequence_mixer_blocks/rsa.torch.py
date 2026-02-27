# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils import divide_if_divisible, is_xma_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import is_glu
from ..convolution import ParameterizedConv1d
from ..decay_gate import SoftplusDecayGate
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution


if is_xma_available():
    from xma import rsa


class RSA(nn.Module):
    def __init__(
        self,
        input_size: int,
        k_head_dim: int,
        v_head_dim: int,
        output_size: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        num_f_heads: int,
        num_g_heads: int,
        num_weight_heads: int,
        use_residual: bool,
        kernel_size: int | None,
        activation_function: str | None,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        use_softplus_decay: bool,
        norm_after_flatten: bool,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
        beta: float = 8.
    ) -> RSA:
        super().__init__()

        self.input_size = input_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.use_residual = use_residual
        self.use_softplus_decay = use_softplus_decay
        self.norm_after_flatten = norm_after_flatten
        self.rnn_size = 1024
        # self.rnn = nn.GRU(d_embedding, d_hidden, num_layers=n_layers, bias=True, batch_first=True, dropout=0.0, bidirectional=False, device=None, dtype=None)
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.rnn_size,
            num_layers=1,
            # dropout=0.1,
            batch_first=True,
            bias=True,
            bidirectional=False,
        )
        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.rnn_size, self.output_size, bias=False, std=std)
        self.h = nn.Parameter(torch.zeros(self.rnn_size))
        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        conv_state, rsa_state = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)
        h = self.h.expand(1, x.size(0), self.rnn_size)
        x, h = self.rnn(input=x, hx=h)
        x = self.output_projection(x)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        self.rnn.weight_hh_l0.data = 0.01 * self.rnn.weight_hh_l0 # /  torch.std(self.rnn.weight_hh_l0)
        # print(list(x for x, _ in self.rnn.named_parameters()))
        # W = torch.eye(self.v_head_dim)
        # W = W[None, ...].expand(self.num_heads, -1, -1)
        # with torch.no_grad():
        #     nn.init.zeros_(self.state_weight)
        #     if not self.state_weight.is_meta:
        #         orig_placements = self.state_weight.placements
        #         device_mesh = self.state_weight.device_mesh
        #         local_weight: torch.Tensor = self.state_weight.to_local()
        #         for i in range(local_weight.size(0)):
        #             nn.init.eye_(local_weight[i])
        #         self.state_weight.redistribute(device_mesh=device_mesh, placements=orig_placements)
        # self.state_weight.copy_(W)

        # if self.use_residual:
        #     nn.init.ones_(self.D)

    # def extra_repr(self) -> str:
    #     return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
