from ..config import MoEDolomiteConfig
from .auxfree import AuxFreeMoE
from .base import SparseMoE
from .scatter import ScatterMoE
from .virouter import VarMoE


_MOE_MODULES = {"eager": SparseMoE, "scattermoe": ScatterMoE, "varouter": VarMoE, "auxfree": AuxFreeMoE}


def get_moe(
    config: MoEDolomiteConfig,
    moe_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> SparseMoE | ScatterMoE:
    if moe_implementation in _MOE_MODULES:
        return _MOE_MODULES[moe_implementation](config, use_padding_free_transformer, layer_idx=layer_idx)

    raise ValueError(f"unexpected `moe_implementation` {moe_implementation}")
