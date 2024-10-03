from ...mixins import CausalLMModelMixin_TP
from .base import MoEDolomiteModel_TP, MoEDolomitePreTrainedModel_TP
from .weights import get_moe_dolomite_tp_state_dict


class MoEDolomiteForCausalLM_TP(MoEDolomitePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = MoEDolomiteModel_TP
    tensor_parallel_state_dict_function = get_moe_dolomite_tp_state_dict
