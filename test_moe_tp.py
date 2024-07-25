import os

import torch
import torch.distributed
from transformers import set_seed
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from dolomite_engine.hf_models.models.moe_dolomite.moe.scatter import _ParameterizedScatteredExperts
from dolomite_engine.hf_models.models.moe_dolomite.moe_TP.scatter import _ColumnParallelScatteredExperts
from dolomite_engine.utils import ProcessGroupManager
import scattermoe

set_seed(42)
tp_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(tensor_parallel_size=tp_size)

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
# cuda_rng_tracker = CUDA_RNGStatesTracker()
# cuda_rng_tracker.add("tensor-parallel-seed", 42)
# set_cuda_rng_tracker(cuda_rng_tracker)
tmp_path = "tmp/"
torch_dtype = torch.bfloat16
num_experts = 8
k = 2
in_features = 1024
out_features = 1024
batch_size = 512
std = 0.02

model = _ParameterizedScatteredExperts(
    num_experts=num_experts,
    input_size=in_features,
    output_size=out_features,
    std=std
)
model.to(torch.cuda.current_device())

if torch.distributed.get_rank() == 0:
    print("Initializing on device 0.")
    weight = model.weight.data
    input_tensor = torch.randn(batch_size, in_features, device=torch.cuda.current_device(), dtype=torch_dtype)
    logits = torch.randn(batch_size, num_experts, device=torch.cuda.current_device(), dtype=torch_dtype)
    expert_p, expert_idxs = torch.topk(logits, k=k)
    expert_idxs = expert_idxs.to(dtype=torch.int32).to(device=torch.cuda.current_device())
else:
    weight = torch.empty((num_experts, out_features, in_features), dtype=torch_dtype, device=torch.cuda.current_device())
    input_tensor = torch.empty((batch_size, in_features), device=torch.cuda.current_device(), dtype=torch_dtype)
    expert_idxs = torch.empty((batch_size, k), device=torch.cuda.current_device(), dtype=torch.int32)


rank = torch.distributed.get_rank()
torch.distributed.broadcast(weight, 0)
torch.distributed.broadcast(input_tensor, 0)
torch.distributed.broadcast(expert_idxs, 0)

model_tp = _ColumnParallelScatteredExperts(
    num_experts=num_experts,
    input_size=in_features,
    output_size=out_features,
    std=std
)


model.load_state_dict({"weight": weight})
weight = weight.view(num_experts, tp_size, -1, in_features)
model_tp.load_state_dict({"weight": weight[:, rank]})
print("Rank", rank, "waiting...")

torch.distributed.barrier()

with torch.no_grad():
    sorted_expert_idxs, sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(expert_idxs)
    padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(
        sorted_expert_idxs, num_experts
    )


# set model to eval mode
model_tp = model_tp.to(torch_dtype)
model_tp.eval()
model.eval()
set_seed(42)

output_tp = model_tp(input_tensor)
output_ref = model(input_tensor)
output_ref_chunk = output_ref.view(batch_size, tp_size, -1)[:, rank]
print((output_tp - output_ref_chunk).abs().max())

ProcessGroupManager.destroy_process_groups()
