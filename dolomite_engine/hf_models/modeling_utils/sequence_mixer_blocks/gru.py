# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence
from .rnn import RNN


if is_cute_kernels_available():
    from cute_kernels import gru_cute, gru_torch


# HACK TUNING CODE
import os


tuning_params = {
    "input_head_norm": "rmsnorm",
    "output_head_norm": "gatednorm",
    "factor_mul": 1,
    "state_weight_init": "orthogonal",
    "gate_in_state_init": "zero",
    "forget_bias_init": "all_heads_gradual_reset",
    "reset_bias_init": 1.0,
}

for key in tuning_params:
    if key in os.environ:
        tuning_params[key] = type(tuning_params[key])(os.environ[key])
    else:
        print(key, "not existent")
# HACK END


class GroupedLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        std: float | None = None,
    ) -> None:
        self.std = std
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.in_dim = divide_if_divisible(in_channels, groups, "in_channels must be divisible by groups")
        self.out_dim = divide_if_divisible(out_channels, groups, "out_channels must be divisible by groups")
        self.weight = nn.Parameter(torch.empty(self.groups, self.in_dim, self.out_dim))
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"groups={self.groups}, in_channels={self.in_channels}, out_channels={self.out_channels}, in_dim={self.in_dim}, out_dim={self.out_dim}"

        # mark_parameter_as_no_weight_decay(self.bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()

    def forward(self, x):
        x_size = x.size()
        # X: ..., groups * in_dim
        x = x.view(-1, self.groups, self.in_dim)
        # X: ..., groups, in_dim
        x = x.transpose(1, 0)
        # X: groups,  *, in_dim
        # weight: groups, in_dim, out_dim
        y = torch.bmm(x, self.weight)
        # y: groups, *, out_dim
        y = y.transpose(1, 0)
        # y: *, groups, out_dim
        y = y.reshape(*(x_size[:-1]), self.out_channels)
        return y


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
        head_activation: str = "tanh",
        factor: float | None = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")

        self.head_group_size = 8
        self.num_groups = divide_if_divisible(self.num_heads, self.head_group_size, "num_heads // num_groups")
        # self.grouped_state_size = self.num_groups * self.state_head_dim
        self.grouped_state_size = self.num_groups * int(self.state_head_dim * 3)



        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        self.input_projection = ParameterizedLinear(
            self.input_size,
            self.grouped_state_size + # input
            self.grouped_state_size + # gate
            self.num_heads +          # forget_gate
            self.num_heads,           # reset_head
            bias=add_bias,
            std=std,
        )

        self.head_input_ln = get_normalization_function("rmsnorm", self.grouped_state_size)
        self.head_input_projection = GroupedLinear(
           in_channels=self.grouped_state_size, out_channels=self.state_size, groups=self.num_groups, std=std
        )
        self.state_weight_std = std
        # self.state_weight = nn.Parameter(torch.empty(3 * self.num_heads, self.state_head_dim, self.state_head_dim))
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

        self.head_output_projection = GroupedLinear(
           in_channels=self.state_size, out_channels=self.grouped_state_size, groups=self.num_groups, std=std
        )



        # if tuning_params['output_head_norm'] == 'rmsnorm':
        #     self.ln_output_head = get_normalization_function("rmsnorm", self.state_size)
        # elif tuning_params['output_head_norm'] == 'groupnorm':
        #     self.ln_output_head = nn.GroupNorm(self.num_heads, self.state_size)
        # elif tuning_params['output_head_norm'] == 'gatednorm':
        #     self.ln_output_head = get_normalization_function("silu_gated_rmsnorm", self.state_size)

        self.ln_output_head = get_normalization_function("silu_gated_rmsnorm", self.grouped_state_size)


        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)


        self.output_projection = ParameterizedLinear(self.grouped_state_size, self.output_size, bias=False, std=std)

        if factor is None:
            factor = tuning_params["factor_mul"]

        self.factor = factor
        self.forget_bias = nn.Parameter(torch.empty(self.num_heads))
        self.reset_bias = nn.Parameter(torch.empty(self.num_heads))
        self.reset_parameters()

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)
        # mark_parameter_as_no_weight_decay(self.state_weight)
        mark_parameter_as_no_weight_decay(self.forget_bias)
        mark_parameter_as_no_weight_decay(self.reset_bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # nn.init.normal_(self.state_weight, std=self.state_weight_std)
        # nn.init.zeros_(self.state_weight)
        # for i in range(self.state_weight.size(0)):
        #     nn.init.orthogonal_(self.state_weight[i])

        nn.init.normal_(self.state_weight, std=self.state_weight_std)

        if tuning_params["state_weight_init"] == "identity":
            self.state_weight.data = (
                self.state_weight.data
                + torch.eye(self.state_head_dim, dtype=self.state_weight.dtype, device=self.state_weight.device)
                / self.factor
            )
        elif tuning_params["state_weight_init"] == "orthogonal":
            nn.init.zeros_(self.state_weight)
            W = torch.empty_like(self.state_weight, device=torch.device("cuda"))
            for i in range(self.state_weight.size(0)):
                nn.init.orthogonal_(W[i])
            self.state_weight.data[:] = W.cpu() / self.factor

        if tuning_params["gate_in_state_init"] == "zero":
            # set the gates to 0 init.
            # assert self.head_projection.weight.size(0) == self.num_head_groups * 3
            # assert self.state_weight.size(0) == self.num_heads * 3
            # nn.init.zeros_(self.head_projection.weight[self.num_head_groups:])
            # nn.init.zeros_(self.state_weight[self.num_heads :])
            pass

        nn.init.zeros_(self.forget_bias)
        nn.init.zeros_(self.reset_bias)
        # if not self.forget_bias.is_meta:
        #     if tuning_params['forget_bias_init'] == 'within_group_gradual':
        #         forget_init = torch.linspace(0.01, 0.99, self.head_group_size)
        #         b = self.forget_bias.data.view(self.num_head_groups, self.head_group_size, self.state_head_dim)
        #         b = b + (torch.log(forget_init) - torch.log(1 - forget_init))[None, :, None]
        #         self.forget_bias.data[:] = b.view(self.forget_bias.size())
        #         nn.init.constant_(self.reset_bias, 1.)
        #     elif tuning_params['forget_bias_init'] == 'all_heads_gradual':
        #         forget_init = torch.linspace(0.01, 0.99, self.num_heads)
        #         b = self.forget_bias.data
        #         b = b + (torch.log(forget_init) - torch.log(1 - forget_init))[:, None]
        #         self.forget_bias.data[:] = b
        nn.init.constant_(self.reset_bias, tuning_params["reset_bias_init"])

    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            batch_size, sequence_length = input.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                input = pack_sequence(inputs=input, cu_seqlens=cu_seqlens)

        input, gate, forget_input, reset_input = self.input_projection(input).split(
            (self.grouped_state_size, self.grouped_state_size, self.num_heads, self.num_heads), dim=-1
        )
        input = self.head_input_ln(input)
        input = self.head_input_projection(input)

        weight = self.state_weight * self.factor
        # weight, forget_weight, reset_weight = weight.chunk(3, dim=0)
        forget_weight = torch.zeros_like(weight)
        reset_weight = forget_weight

        input: torch.Tensor = input * self.factor
        # input = input.view(*(input.size()[:-1]), self.num_groups, self.state_head_dim)
        # input = input.repeat_interleave(self.head_group_size, dim=-2)
        input = input.view(*(input.size()[:-1]), self.num_heads, self.state_head_dim)
        forget_input: torch.Tensor = forget_input + self.forget_bias
        forget_input = forget_input.unsqueeze(-1).expand_as(input).contiguous()
        reset_input: torch.Tensor = reset_input + self.reset_bias
        reset_input = reset_input.unsqueeze(-1).expand_as(input).contiguous()

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)

        input = (gru_cute if is_kernel_allowed(Kernel.gru_cute) else gru_torch)(
            input=input,
            weight=weight,
            forget_input=forget_input,
            forget_weight=forget_weight,
            reset_input=reset_input,
            reset_weight=reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(
                inputs=input, cu_seqlens=cu_seqlens, desired_shape=(batch_size, sequence_length, *input.size()[1:])
            )

        if cache_params is not None:
            input_state = input[:, -1].view(input.size(0), -1)
            cache_params.update(state=input_state, num_tokens_added=input.size(1), layer_idx=self.layer_idx)
        # input = self.output_head_projection(input.flatten(-2, -1))
        input = input.flatten(-2, -1)
        input = self.head_output_projection(input)
        input = self.ln_output_head(input, gate)
        input = self.output_projection(input)

        return input

    def extra_repr(self) -> str:
        return f"gradient_clipping={self.gradient_clipping}, weight_shape={str(tuple(self.state_weight.shape))}, factor={self.factor}, head_group_size={self.head_group_size},"
