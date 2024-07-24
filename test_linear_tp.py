import os

import torch
import torch.distributed
from transformers import set_seed

from dolomite_engine.hf_models.modeling_utils.linear import ParameterizedLinear
from dolomite_engine.hf_models.modeling_utils_TP.linear import ColumnParallelLinear
from dolomite_engine.utils import ProcessGroupManager


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
in_features = 1024
out_features = 8192
batch_size = 512

model = ParameterizedLinear(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    device=torch.cuda.current_device(),
    dtype=torch_dtype,
    std=0.02,
)

if torch.distributed.get_rank() == 0:
    print("Initializing on device 0.")
    weight = model.weight.data
    input_tensor = torch.randn(batch_size, in_features, device=torch.cuda.current_device(), dtype=torch_dtype)
else:
    weight = torch.empty((out_features, in_features), dtype=torch_dtype, device=torch.cuda.current_device())
    input_tensor = torch.empty((batch_size, in_features), device=torch.cuda.current_device(), dtype=torch_dtype)

torch.distributed.broadcast(weight, 0)
torch.distributed.broadcast(input_tensor, 0)

model_tp = ColumnParallelLinear(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    device=torch.device("cuda"),
    dtype=torch_dtype,
    std=0.02,
    use_padding_free_transformer=False,
    sequence_parallel=False,
)
rank = torch.distributed.get_rank()

model.load_state_dict({"weight": weight})

weight = weight.view(tp_size, -1, in_features)
model_tp.load_state_dict({"weight": weight[rank]})

torch.distributed.barrier()

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
