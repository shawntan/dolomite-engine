import os

import scattermoe
import torch
import torch.distributed
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import set_seed

from dolomite_engine.hf_models.models.moe_dolomite.config import MoEDolomiteConfig
from dolomite_engine.hf_models.models.moe_dolomite.moe.scatter import ParameterizedScatteredExperts, ScatterMoE
from dolomite_engine.hf_models.models.moe_dolomite_TP.moe_TP.scatter import RowParallelScatteredExperts
from dolomite_engine.utils import ProcessGroupManager


set_seed(42)
tp_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(tensor_parallel_size=tp_size)

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
# cuda_rng_tracker = CUDA_RNGStatesTracker()
# cuda_rng_tracker.add("tensor-parallel-seed", 42)
# set_cuda_rng_tracker(cuda_rng_tracker)
config = MoEDolomiteConfig(add_bias=False)
torch_dtype = torch.float32
# num_experts = 8
# k = 2
# in_features = 1024
# out_features = 1024
# batch_size = 512
local_moe = ScatterMoE(config, use_padding_free_transformer=False, layer_idx=0)
print(local_moe)
rank = torch.distributed.get_rank()
if rank == 0:
    print("Initializing on device ", rank)
    weight = model.weight.data
    input_tensor = torch.randn(k * batch_size, in_features, device=torch.cuda.current_device(), dtype=torch_dtype)
    logits = torch.randn(batch_size, num_experts, device=torch.cuda.current_device(), dtype=torch_dtype)
    expert_p, expert_idxs = torch.topk(torch.softmax(logits, dim=-1), k=k)
    expert_idxs = expert_idxs.to(dtype=torch.int32).to(device=torch.cuda.current_device())
else:
    weight = torch.empty(
        (num_experts, out_features, in_features), dtype=torch_dtype, device=torch.cuda.current_device()
    )
    input_tensor = torch.empty((k * batch_size, in_features), device=torch.cuda.current_device(), dtype=torch_dtype)
    expert_idxs = torch.empty((batch_size, k), device=torch.cuda.current_device(), dtype=torch.int32)
    expert_p = torch.empty((batch_size, k), device=torch.cuda.current_device(), dtype=torch_dtype)


torch.distributed.broadcast(weight, 0)
torch.distributed.broadcast(input_tensor, 0)
torch.distributed.broadcast(expert_idxs, 0)
torch.distributed.broadcast(expert_p, 0)

model_tp = _RowParallelScatteredExperts(
    num_experts=num_experts,
    input_size=in_features,
    output_size=out_features,
    std=std,
    device=torch.device(torch.cuda.current_device()),
    dtype=torch_dtype,
)

model.load_state_dict({"weight": weight})
weight = weight.view(num_experts, out_features, tp_size, -1)
model_tp.load_state_dict({"weight": weight[..., rank, :]})

torch.distributed.barrier()

with torch.no_grad():
    sorted_expert_idxs, sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(expert_idxs)
    padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(sorted_expert_idxs, num_experts)


# set model to eval mode
model_tp = model_tp.to(torch_dtype)
model_tp.eval()
model.eval()
set_seed(42)

output_tp = model_tp(
    input_tensor.view(k * batch_size, tp_size, -1)[:, rank],
    1,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    expert_offsets,
    gates=expert_p,
    grouped_in=True,
    grouped_out=False,
)

output_ref = model(
    input_tensor,
    1,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    expert_offsets,
    gates=expert_p,
    grouped_in=True,
    grouped_out=False,
)
if rank == 0:
    print(output_tp.size())
    print(output_ref.size())
print((output_tp - output_ref).abs().max())

ProcessGroupManager.destroy_process_groups()
