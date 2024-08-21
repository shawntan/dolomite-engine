import os

import scattermoe
import torch
import torch.distributed
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import set_seed

from dolomite_engine.hf_models.models.moe_dolomite.config import MoEDolomiteConfig
from dolomite_engine.hf_models.models.moe_dolomite.moe.scatter import ParameterizedScatteredExperts, ScatterMoE
from dolomite_engine.hf_models.models.moe_dolomite_TP.moe_TP.scatter import ScatterMoETP
from dolomite_engine.utils import ProcessGroupManager


set_seed(42)
tp_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(tensor_parallel_size=tp_size)

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
# cuda_rng_tracker = CUDA_RNGStatesTracker()
# cuda_rng_tracker.add("tensor-parallel-seed", 42)
# set_cuda_rng_tracker(cuda_rng_tracker)
config = MoEDolomiteConfig(activation_function="swiglu", add_bias=False)
torch_dtype = torch.float32
# num_experts = 8
# k = 2
# in_features = 1024
# out_features = 1024
batch_size = 16
rank = torch.distributed.get_rank()
local_moe = ScatterMoE(config, use_padding_free_transformer=True, layer_idx=0)
shard_moe = ScatterMoETP(
    config, device=torch.cuda.current_device(), dtype=torch_dtype, use_padding_free_transformer=True, layer_idx=0
)

input_tensor = torch.randn(batch_size, config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype)
gate_weight = local_moe.gate.weight
c_fc_weight = local_moe.c_fc.weight
c_proj_weight = local_moe.c_proj.weight

torch.distributed.broadcast(input_tensor, 0)
torch.distributed.broadcast(gate_weight, 0)
torch.distributed.broadcast(c_fc_weight, 0)
torch.distributed.broadcast(c_proj_weight, 0)
if rank == 0:
    print("Rank", rank)
    print(local_moe)
    print([(n, p.size()) for n, p in local_moe.named_parameters()])
    print(shard_moe)
    print([(n, p.size()) for n, p in local_moe.named_parameters()])

if rank == 0:
    print("Distributing local_moe params...")
local_moe.load_state_dict({"gate.weight": gate_weight, "c_fc.weight": c_fc_weight, "c_proj.weight": c_proj_weight})
torch.distributed.barrier()

if rank == 0:
    print("Distributing sharded_moe params...")
c_fc_1_weight, c_fc_2_weight = c_fc_weight.chunk(2, dim=1)
sharded_inter_dim = shard_moe.c_proj.in_features_per_device
shard_moe.load_state_dict(
    {
        "gate.weight": gate_weight,
        "c_fc.weight": torch.cat(
            (
                c_fc_1_weight[:, sharded_inter_dim * rank : (rank + 1) * sharded_inter_dim, :],
                c_fc_2_weight[:, sharded_inter_dim * rank : (rank + 1) * sharded_inter_dim, :],
            ),
            dim=1,
        ),
        "c_proj.weight": c_proj_weight[:, :, sharded_inter_dim * rank : (rank + 1) * sharded_inter_dim],
    }
)
torch.distributed.barrier()

shard_moe(input_tensor)

ProcessGroupManager.destroy_process_groups()
exit()
model.load_state_dict({"weight": weight})
weight = weight.view(num_experts, out_features, tp_size, -1)
model_tp.load_state_dict({"weight": weight[..., rank, :]})


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
