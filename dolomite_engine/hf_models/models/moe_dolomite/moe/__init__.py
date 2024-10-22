from ..config import MoEDolomiteConfig
from .base import SparseMoE
from .scatter import ScatterExperts


_MOE_MODULES = {"eager": SparseMoE, "scattermoe": ScatterExperts}


def get_moe(
    config: MoEDolomiteConfig,
    moe_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> SparseMoE | ScatterExperts:
    if moe_implementation in _MOE_MODULES:
        return _MOE_MODULES[moe_implementation](config, use_padding_free_transformer, layer_idx=layer_idx)

    raise ValueError(f"unexpected `moe_implementation` {moe_implementation}")
