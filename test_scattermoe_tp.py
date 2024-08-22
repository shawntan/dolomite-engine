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
config = MoEDolomiteConfig(n_embd=2048, add_bias=False, embd_pdrop=0.0, resid_pdrop=0.0)
torch_dtype = torch.bfloat16
batch_size = 512
rank = torch.distributed.get_rank()
local_moe = ScatterMoE(config, use_padding_free_transformer=True, layer_idx=0)
local_moe = local_moe.to(device=torch.cuda.current_device(), dtype=torch_dtype)
shard_moe = ScatterMoETP(
    config, device=torch.cuda.current_device(), dtype=torch_dtype, use_padding_free_transformer=True, layer_idx=0
)
input_tensor = 0.02 * torch.randn(
    batch_size, config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype, requires_grad=True
)
gate_weight = 0.02 * torch.randn_like(local_moe.gate.weight, requires_grad=True)
c_fc_weight = 0.02 * torch.randn_like(local_moe.c_fc.weight, requires_grad=True)
c_proj_weight = 0.02 * torch.randn_like(local_moe.c_proj.weight, requires_grad=True)

grad_tensor = 0.02 * torch.randn(batch_size, config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype)

torch.distributed.broadcast(input_tensor, 0)
torch.distributed.broadcast(gate_weight, 0)
torch.distributed.broadcast(c_fc_weight, 0)
torch.distributed.broadcast(c_proj_weight, 0)
torch.distributed.broadcast(grad_tensor, 0)

if rank == 0:
    print("Rank", rank)
    print(config)
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

shard_moe.gate.load_state_dict({"weight": gate_weight})
if False:
    sharded_inter_dim = shard_moe.c_proj.in_features_per_device
    c_fc_1_weight, c_fc_2_weight = c_fc_weight.chunk(2, dim=1)
    shard_moe.c_fc.load_state_dict(
        {
            "weight": torch.cat(
                (
                    c_fc_1_weight[:, sharded_inter_dim * rank : (rank + 1) * sharded_inter_dim, :],
                    c_fc_2_weight[:, sharded_inter_dim * rank : (rank + 1) * sharded_inter_dim, :],
                ),
                dim=1,
            )
        }
    )
else:
    shard_moe.c_fc.load_state_dict(
        {"weight": c_fc_weight.view(c_fc_weight.size(0), tp_size, -1, c_fc_weight.size(2))[:, rank]}
    )
shard_moe.c_proj.load_state_dict(
    {"weight": c_proj_weight.view(c_proj_weight.size(0), c_proj_weight.size(1), tp_size, -1)[:, :, rank]}
)

torch.distributed.barrier()
local_input_tensor = input_tensor
shard_input_tensor = input_tensor.clone()

local_out, local_logits = local_moe(local_input_tensor)
shard_out, shard_logits = shard_moe(shard_input_tensor)
local_input_tensor_grad, local_gate_weight_grad = torch.autograd.grad(
    outputs=(local_out), inputs=(local_input_tensor, local_moe.gate.weight), grad_outputs=(grad_tensor,)
)
shard_input_tensor_grad, shard_gate_weight_grad = torch.autograd.grad(
    outputs=(shard_out), inputs=(shard_input_tensor, shard_moe.gate.weight), grad_outputs=(grad_tensor,)
)

if rank == 0:
    print()
    print("logits error:")
for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_logits - shard_logits).abs().max())
    torch.distributed.barrier()


if rank == 0:
    print()
    print("out error:")
for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_out - shard_out).abs().max())
    torch.distributed.barrier()

if rank == 0:
    print()
    print("input grad error:")
for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_input_tensor_grad - shard_input_tensor_grad).abs().max())
    torch.distributed.barrier()

if rank == 0:
    print()
    print("gate grad error:")
for r in range(tp_size):
    if rank == r:
        print(shard_gate_weight_grad)
        # print("Rank %d:" % r, (local_gate_weight_grad - shard_gate_weight_grad).abs().max())
    torch.distributed.barrier()


ProcessGroupManager.destroy_process_groups()
