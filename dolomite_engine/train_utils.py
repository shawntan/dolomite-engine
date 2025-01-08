import logging

import torch
from torch.distributed import ReduceOp
from transformers import AutoConfig

from .enums import GradientCheckpointingMethod
from .hf_models import is_custom_model
from .hf_models.modeling_utils import is_glu
from .utils import ExperimentsTracker, MetricsTrackingDict, ProcessGroupManager, log_metrics


def all_reduce_metrics_tracker(metrics_tracker: MetricsTrackingDict) -> MetricsTrackingDict:
    tensor = [metrics_tracker[key] for key in metrics_tracker]
    tensor = torch.stack(tensor)
    # NOTE the cpu() call was to save memory but might not be needed anymore
    # tensor = torch.stack(tensor) / ProcessGroupManager.get_data_parallel_world_size()
    # tensor = tensor.cpu()
    # gloo op doesn't support averaging so we do sum and divide by world size above
    torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())
    tensor = tensor.tolist()

    for i, key in enumerate(metrics_tracker):
        metrics_tracker[key] = tensor[i]

    return metrics_tracker


def track_metrics(
    global_step: int, experiments_tracker: ExperimentsTracker, metrics_tracker: MetricsTrackingDict, context: str
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        metrics_tracker (float): metrics tracker
        context (str): experiment context
    """

    # experiments tracker
    experiments_tracker.track(metrics_tracker.get_dict(), step=global_step, context=context)

    message = f"step = {global_step}"
    for key in metrics_tracker:
        if key == "learning_rate":
            message += f", {key} = {metrics_tracker[key]:.4e}"
        else:
            message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

    log_metrics(logging.INFO, message)


def get_torch_profiler(torch_profiler_trace_path: str) -> torch.profiler.profile:
    torch_profiler = None
    if torch_profiler_trace_path is not None:
        torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profiler_trace_path),
            record_shapes=True,
        )

    return torch_profiler


def get_model_tflops(
    config: AutoConfig,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing_method: GradientCheckpointingMethod | None,
    gradient_checkpointing_args: dict,
) -> None:
    if not is_custom_model(config.model_type):
        return 0

    b = batch_size
    s = sequence_length
    h = config.n_embd
    f = config.n_inner
    n = config.n_head
    k = config.num_key_value_heads
    l = config.n_layer
    v = config.vocab_size

    mlp_flops = 4 * b * s * h * f
    if config.model_type == "moe_dolomite":
        mlp_flops *= config.num_experts_per_tok

    if is_glu(config.activation_function):
        mlp_flops *= 1.5

    attention_flops = 4 * b * s * h * (h * (1 + k / n) + s)

    forward_flops = attention_flops + mlp_flops

    if gradient_checkpointing_method == GradientCheckpointingMethod.block:
        num_layers_checkpointed = gradient_checkpointing_args.get("num_blocks", l)
        fraction_of_layers_checkpointed = num_layers_checkpointed / l
        backward_flops = (2 + fraction_of_layers_checkpointed) * forward_flops
    else:
        backward_flops = 2 * forward_flops

    model_flops = l * (forward_flops + backward_flops)
    model_flops += 6 * b * s * h * v
    model_flops /= 10**12

    return model_flops
