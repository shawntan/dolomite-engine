import os

import torch
import torch.distributed
from transformers import set_seed

from dolomite_engine.hf_models import AttentionHeadType, GPTDolomiteConfig, GPTDolomiteForCausalLM_TP
from dolomite_engine.hf_models.modeling_utils_TP.linear import ColumnParallelLinear
from dolomite_engine.utils import (
    CUDA_RNGStatesTracker,
    ProcessGroupManager,
    SafeTensorsWeightsManager,
    set_cuda_rng_tracker,
    string_to_torch_dtype,
)


set_seed(42)

ProcessGroupManager(tensor_parallel_size=int(os.getenv("WORLD_SIZE")))

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
cuda_rng_tracker = CUDA_RNGStatesTracker()
cuda_rng_tracker.add("tensor-parallel-seed", 42)
set_cuda_rng_tracker(cuda_rng_tracker)

torch_dtype = torch.bfloat16
model_tp = ColumnParallelLinear(
    in_features=1024,
    out_features=8192,
    bias=False,
    device=torch.device("cuda"),
    dtype=torch_dtype,
    std=0.02,
    use_padding_free_transformer=False,
    sequence_parallel=False,
)
torch.distributed.barrier()


# copy to device without copying storage
model_tp = model_tp.to_empty(device=torch.cuda.current_device())

# load weights into tensor parallel model using SafeTensorsWeightsManager class
# this avoids loading multiple copies of the parameters in CPU memory
safetensors_weight_manager = SafeTensorsWeightsManager("tmp/")
model_tp.load_from_safetensors_weights_manager(safetensors_weight_manager)

# set model to eval mode
model_tp = model_tp.to(torch_dtype)
model_tp.eval()

set_seed(42)

input_tensor = torch.randn(512, 1024, device=torch.cuda.current_device())
output_tp = model_tp(input_tensor)
exit()
if torch.distributed.get_rank() == 0:
    output = model(input_ids=input_ids, labels=labels)
    loss = output[0]
    logits = output[1]

    if args.use_padding_free_transformer:
        logits_tp = logits_tp.reshape(batch_size, sequence_length, -1)

    error = (logits - logits_tp).abs().max()
    assert error < 5e-4, "logits don't match for normal and tensor parallel model"

    error = (loss - loss_tp).abs().max()
    assert error < 3e-6, "losses don't match for normal and tensor parallel model"

ProcessGroupManager.destroy_process_groups()
