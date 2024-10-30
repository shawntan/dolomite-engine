import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate

from .....distributed import dtensor_to_tensor, tensor_to_dtensor
from .....utils import ProcessGroupManager
from ...dtensor_module import DTensorModule
from ...TP import get_module_placements


class LayerNorm_TP(nn.LayerNorm, DTensorModule):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                current_placement=Replicate(),
            )
        )
        self.bias = nn.Parameter(
            tensor_to_dtensor(
                self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Replicate()
            )
        )

        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=self.placement)
        input = super().forward(input)
        input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=self.placement)
        return input
