from copy import deepcopy

import torch
import torch.nn as nn
from transformers import DynamicCache

from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.mlp import MLP
from ..moe_dolomite.moe.scatter import ScatterMoE
from .config import MoEDolomiteConfig


class SUTMoE(ScatterMoE):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        hidden_states = self._compute_experts(hidden_states, router_weights, selected_experts)

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        freq = selected_experts.flatten().bincount(minlength=self.num_experts).to(dtype=router_logits.dtype)
        return hidden_states, router_logits, freq


class SUTBlock(nn.Module):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        self.moe = SUTMoE(config, use_padding_free_transformer, layer_idx)

        self.mlp = None
        if config.shared_n_inner is not None:
            shared_config = deepcopy(config)
            shared_config.n_inner = config.shared_n_inner
            self.mlp = MLP(shared_config)
            del shared_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        output_router_logits: bool = False,
        output_aux_loss: bool = True,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        # residual connection
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states, router_logits, freq = self._compute_moe_and_mlp(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        # residual connection
        hidden_states = hidden_states + residual
        """
        outputs = (hidden_states,)
        if output_router_logits:
            outputs += (router_logits,)
        if output_aux_loss:
            outputs += (aux_loss,)
        """
        return hidden_states, router_logits, freq

    def _compute_moe_and_mlp(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        moe_output, router_logits, aux_loss = self.moe(hidden_states)

        if self.mlp is not None:
            mlp_output = self.mlp(hidden_states)
            moe_output = mlp_output + moe_output

        return moe_output, router_logits, aux_loss
