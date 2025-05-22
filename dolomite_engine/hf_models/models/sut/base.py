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
from ...utils import convert_padding_free_lists_to_tensors, is_generation_cache_enabled
from . import layer
from .config import SUTConfig
from .layer import SUTBlock


class SUTPreTrainedModel(PreTrainedModelMixin):
    config_class = SUTConfig
    layer_class = SUTBlock
    _no_split_modules = ["SUTBlock"]


class SUTModel(SUTPreTrainedModel, BaseModelMixin):
    def _init_model(self, config: SUTConfig, **kwargs) -> None:
        self.embed_dim = config.hidden_size
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.embedding_dropout = (
            nn.Identity() if config.embedding_dropout == 0 else nn.Dropout(config.embedding_dropout)
        )
        if len(config.sequence_mixer_blocks) == 1:
            self.sequence_mixer_block_types = [config.sequence_mixer_blocks[0].sequence_mixer_type]
            self.h = nn.ModuleList(
                [self.layer_class(config, use_padding_free_transformer=self.use_padding_free_transformer, layer_idx=0)]
            )
        elif len(config.sequence_mixer_blocks) == 3:
            self.h_first = Block(config, use_padding_free_transformer=self.use_padding_free_transformer, layer_idx=0)
            self.sequence_mixer_block_types = [config.sequence_mixer_blocks[1].sequence_mixer_type]
            self.h = nn.ModuleList(
                [self.layer_class(config, use_padding_free_transformer=self.use_padding_free_transformer, layer_idx=1)]
            )
            self.h_last = Block(
                config,
                use_padding_free_transformer=self.use_padding_free_transformer,
                layer_idx=len(config.sequence_mixer_blocks) - 1,
            )

        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.rope_dim = config.rope_dim

        self.position_embedding_type = config.position_embedding_type
        self._setup_positional_encoding()
        self.num_iters = self.config.num_iters

        self.num_forward_count = 0
        self.num_steps_tick = 10 * 20
        self.curr_iters = self.config.num_iters
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
        mamba_mask_computed = False
        block: SUTBlock = self.h[0]
        sequence_mixer_type = self.sequence_mixer_block_types[0]
        acc_mlp_router_stats = None
        acc_attn_router_stats = None

        if self.h_first is not None:
            hidden_states = self.h_first(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        for i in range(self.num_iters):

            is_mamba_layer = sequence_mixer_type in ["mamba2", "rnn"]

            if is_mamba_layer and not mamba_mask_computed:
                mamba_mask = self._get_mamba_mask(attention_mask, past_key_values)
                mamba_mask_computed = True

            # prev_hidden_states = hidden_states
            hidden_states, mlp_router_stats, attn_router_stats = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=mamba_mask if is_mamba_layer else causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                layer_idx=i,
            )

            acc_mlp_router_stats = layer._update_statistics(acc_mlp_router_stats, mlp_router_stats)
            acc_attn_router_stats = layer._update_statistics(acc_attn_router_stats, attn_router_stats)

            # if (self.curr_iters < self.num_iters) and (i >= self.curr_iters):
            #     hidden_states = 0.999 * prev_hidden_states + 0.001 * hidden_states

        # if self.curr_iters < self.num_iters:
        #     if self.num_forward_count < self.num_steps_tick * self.num_iters:
        #         self.num_forward_count += 1
        #         if self.num_forward_count % self.num_steps_tick == 0:
        #             self.curr_iters = min(self.num_iters, self.curr_iters + 1)
        #             log_rank_0(logging.INFO, f"Increasing num_iters to {self.curr_iters}")

        add_aux_loss(layer._compute_switch_loss(acc_mlp_router_stats))
        add_aux_loss(layer._compute_switch_loss(acc_attn_router_stats))
        if self.h_last is not None:
            hidden_states = self.h_last(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        hidden_states = self.ln_f(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
