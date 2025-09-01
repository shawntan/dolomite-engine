import math

import torch
import torch.nn as nn
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F
from transformers import DynamicCache

from ....utils import ProcessGroupManager, divide_if_divisible, log_rank_0
from ...cache import GenerationCache
from ...config import CommonConfig
from ...loss import add_aux_loss
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from ...mixins.dense import Block
from ...mixins.modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from ...utils import convert_padding_free_lists_to_tensors, is_generation_cache_enabled
from . import layer
from .config import SUTConfig
from .halting import HaltingGate
from .layer import SUTBlock


class SUTPreTrainedModel(PreTrainedModelMixin):
    config_class = SUTConfig
    layer_class = SUTBlock
    _no_split_modules = ["Block"]


class SUTModel(SUTPreTrainedModel, BaseModelMixin):
    def _init_model(self, config: SUTConfig, **kwargs) -> None:
        self.embed_dim = config.hidden_size
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.num_iters = self.config.num_iters

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.embedding_dropout = (
            nn.Identity() if config.embedding_dropout == 0 else nn.Dropout(config.embedding_dropout)
        )

        if config.halting:
            std = _get_std_for_linear(config.initializer_range, config.init_method, config.m_width)
            self.halt = HaltingGate(self.embed_dim, std=std)
        else:
            self.halt = None

        self.sequence_mixer_block_types = [x.sequence_mixer_type for x in config.sequence_mixer_blocks]
        self.enc_layers, self.uni_layers, self.dec_layers = config.enc_uni_dec_layers

        mod_list = []
        idx = 0
        for _ in range(self.enc_layers):
            mod_list.append(
                Block(config, use_padding_free_transformer=self.use_padding_free_transformer, layer_idx=idx)
            )
            idx += 1

        # block_config = copy.deepcopy(config)
        # block_config.m_width = config.m_width * math.sqrt(config.num_iters)
        for _ in range(self.uni_layers):
            sut_block = SUTBlock(
                config,
                use_padding_free_transformer=self.use_padding_free_transformer,
                layer_idx=idx,
                num_iters=self.num_iters,
            )
            mod_list.append(sut_block)
            for p in sut_block.parameters():
                if hasattr(p, "_has_mup_learning_rate") and p._has_mup_learning_rate:
                    p._has_mup_learning_rate = math.sqrt(config.num_iters)
            idx += 1

        for _ in range(self.dec_layers):
            mod_list.append(
                Block(config, use_padding_free_transformer=self.use_padding_free_transformer, layer_idx=idx)
            )
            idx += 1
        self.h = nn.ModuleList(mod_list)
        # for m in self.h:
        #     setattr(m, 'visited', False)

        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )
        self.rope_dim = config.rope_dim

        self.position_embedding_type = config.position_embedding_type
        self._setup_positional_encoding()

        self.full_stack = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> BaseModelOutputWithPast:
        (
            use_cache,
            hidden_states,
            causal_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # ==========================================================================================
        # padding_free:
        #     attention_mask -> None
        # flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values
            )

        mamba_mask = None
        sequence_mixer_type = self.sequence_mixer_block_types[0]
        mamba_mask = self._get_mamba_mask(attention_mask, past_key_values)

        block_idx = 0
        for _ in range(self.enc_layers):
            hidden_states = self.execute_block(
                self.h[block_idx],
                hidden_states,
                past_key_values,
                attention_mask,
                cu_seqlens,
                max_seqlen,
                causal_mask,
                rope_cos_sin,
                mamba_mask,
                sequence_mixer_type,
            )

            block_idx += 1

        # TODO update if multiple UT blocks
        halt_state = None
        kv_hidden_states = None
        acc_mlp_router_stats = [None] * self.uni_layers
        acc_attn_router_stats = [None] * self.uni_layers
        for _ in range(self.num_iters):
            prev_hidden_states = hidden_states
            for u_block_idx in range(self.uni_layers):
                block = self.h[block_idx + u_block_idx]
                is_mamba_layer = sequence_mixer_type in ["mamba2", "rnn"]
                hidden_states, mlp_router_stats, attn_router_stats = block(
                    hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=mamba_mask if is_mamba_layer else causal_mask,
                    rope_cos_sin=rope_cos_sin,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    layer_idx=block_idx + u_block_idx,
                    kv_hidden_states=kv_hidden_states,
                )
                if acc_mlp_router_stats is not None:
                    acc_mlp_router_stats[u_block_idx] = layer._update_statistics(
                        acc_mlp_router_stats[u_block_idx], mlp_router_stats
                    )
                if attn_router_stats is not None:
                    acc_attn_router_stats[u_block_idx] = layer._update_statistics(
                        acc_attn_router_stats[u_block_idx], attn_router_stats
                    )
                # block.visited = True

            if self.halt is not None:
                kv_hidden_states, halt_state = self.halt.forward(prev_hidden_states, hidden_states, halt_state)

        if self.halt is not None:
            hidden_states = kv_hidden_states
        for u_block_idx in range(self.uni_layers):
            if acc_mlp_router_stats[u_block_idx] is not None:
                add_aux_loss(layer._compute_switch_loss(acc_mlp_router_stats[u_block_idx], moe_id=u_block_idx))
        for u_block_idx in range(self.uni_layers):
            if acc_attn_router_stats[u_block_idx] is not None:
                add_aux_loss(layer._compute_switch_loss(acc_attn_router_stats[u_block_idx], moe_id=u_block_idx))

        block_idx = block_idx + self.uni_layers

        for _ in range(self.enc_layers):
            hidden_states = self.execute_block(
                self.h[block_idx],
                hidden_states,
                past_key_values,
                attention_mask,
                cu_seqlens,
                max_seqlen,
                causal_mask,
                rope_cos_sin,
                mamba_mask,
                sequence_mixer_type,
            )
            block_idx += 1

        # assert all(m.visited for m in self.h)
        hidden_states = self.ln_f(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def execute_block(
        self,
        block,
        hidden_states,
        past_key_values,
        attention_mask,
        cu_seqlens,
        max_seqlen,
        causal_mask,
        rope_cos_sin,
        mamba_mask,
        sequence_mixer_type,
    ):
        is_mamba_layer = sequence_mixer_type in ["mamba2", "rnn"]
        hidden_states = block(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=mamba_mask if is_mamba_layer else attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # block.visited = True

        return hidden_states
