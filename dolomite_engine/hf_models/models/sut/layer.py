import torch
import torch.nn as nn
from transformers import DynamicCache

from ...mixins.dense.layer import Block


# from ...modeling_utils import get_attention_module, get_normalization_function
# from ..gpt_dolomite.mlp import MLP
# from ..moe_dolomite.moe.scatter import ScatterMoE
# from .config import MoEDolomiteConfig
class SUTBlock(Block): ...
