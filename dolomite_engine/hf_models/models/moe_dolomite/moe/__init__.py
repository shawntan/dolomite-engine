from ..config import MoEDolomiteConfig
from .base import Experts
from .scatter import ScatterExperts
from .base import MoEMLP


_MOE_MODULES = {"eager": Experts, "scattermoe": ScatterExperts}


def get_moe(
    config: MoEDolomiteConfig,
    moe_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> Experts:
    if moe_implementation in _MOE_MODULES:
        return _MOE_MODULES[moe_implementation](config, use_padding_free_transformer, layer_idx=layer_idx)

    raise ValueError(f"unexpected `moe_implementation` {moe_implementation}")
