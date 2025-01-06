from ...mixins import CausalLMModelMixin
from .base import StickBreakingModel, StickBreakingPreTrainedModel
import torch
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast



class StickBreakingForCausalLM(StickBreakingPreTrainedModel, CausalLMModelMixin):
    base_model_class = StickBreakingModel
    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | list[list[int]] | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> CausalLMOutputWithPast:
        needs_eval_fix = self._use_padding_free_transformer and cu_seqlens is None 
        if needs_eval_fix:
            assert input_ids.size(0) == 1, input_ids.size()
            cu_seqlens = torch.tensor([0, input_ids.size(1)],
                                      device=input_ids.device,
                                      dtype=input_ids.dtype)
            max_seqlen = input_ids.size(1)
            position_ids = torch.arange(max_seqlen, device=input_ids.device, dtype=input_ids.dtype)
            input_ids = input_ids[0]

        outputs = super().forward(
            input_ids, past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cu_seqlens,
            max_seqlen,
            reduction,
        )
        if needs_eval_fix:
            outputs.logits = outputs.logits[None, :]
        return outputs
