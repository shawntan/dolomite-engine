from ...mixins import CausalLMModelMixin
from .base import SUTModel, SUTPreTrainedModel


class SUTForCausalLM(SUTPreTrainedModel, CausalLMModelMixin):
    base_model_class = SUTModel
