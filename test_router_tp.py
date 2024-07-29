import os

import scattermoe
import torch
import torch.distributed
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import set_seed

from dolomite_engine.utils import ProcessGroupManager

from dolomite_engine.hf_models.models.moe_dolomite.moe_TP.scatter import ReplicatedParallelLinear
from dolomite_engine.hf_models.modeling_utils import ParameterizedLinear


set_seed(42)
tp_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(tensor_parallel_size=tp_size)

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
# cuda_rng_tracker = CUDA_RNGStatesTracker()
# cuda_rng_tracker.add("tensor-parallel-seed", 42)
# set_cuda_rng_tracker(cuda_rng_tracker)
tmp_path = "tmp/"
torch_dtype = torch.float32
num_experts = 8
k = 2
in_features = 1024
batch_size = 512
std = 0.02

model = ParameterizedLinear(
    in_features=in_features,
    out_features=num_experts,
    bias=False,
    device=torch.cuda.current_device(),
    dtype=torch_dtype,
    std=std,
)

rank = torch.distributed.get_rank()
input_tensor = torch.randn(
    batch_size, in_features,
    device=torch.cuda.current_device(),
    dtype=torch_dtype,
    requires_grad=True
)
weight = model.weight.data

torch.distributed.broadcast(weight, 0)
torch.distributed.broadcast(input_tensor, 0)

model_tp = ReplicatedParallelLinear(
    in_features=in_features,
    out_features=num_experts,
    device=torch.cuda.current_device(),
    dtype=torch_dtype,
    std=std,
)
model.load_state_dict({"weight": weight})
model_tp.load_state_dict({"weight": weight})


# set model to eval mode
model_tp.eval()
model.eval()

torch.distributed.barrier()
output_tp = model_tp(input_tensor)
output_ref = model(input_tensor)

print("Output on rank", rank, (output_tp - output_ref).abs().max())

set_seed(rank)
grad = torch.rand_like(output_tp)
print(rank, grad[0, 0])
torch.distributed.barrier()
output_tp.backward(grad)
print(model_tp.weight.grad)
ProcessGroupManager.destroy_process_groups()
