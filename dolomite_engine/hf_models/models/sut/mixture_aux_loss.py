import torch
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

from dolomite_engine.utils import ProcessGroupManager

from ....utils.tracking import is_tracking_rank, wandb


moe_stats = {}


def update_moe_stats(moe_id, freq):
    if is_tracking_rank():
        global moe_stats
        freq = freq.clone()
        if moe_id not in moe_stats:
            moe_stats[moe_id] = freq
        else:
            moe_stats[moe_id] += freq


def log_moe_stats(logger, context):
    import numpy as np

    if is_tracking_rank():
        global moe_stats
        # values = {f"{context}/{k}": v for k, v in values.items()}
        values = {}
        for moe_id in moe_stats:
            acc_freq_ = moe_stats[moe_id].float()
            acc_freq = F.normalize(acc_freq_, p=1, dim=0).cpu().numpy()
            idxs = np.arange(acc_freq.shape[0] + 1)
            values[f"{context}/expert-{moe_id}"] = wandb.Histogram(np_histogram=(acc_freq, idxs))
            values[f"{context}/expert-{moe_id}-min"] = acc_freq.min()
            moe_stats[moe_id] = acc_freq_ * 0
        logger.log(values)


_current_id = None
_stats_tracker = None


def _update_statistics(acc_stats, stats):
    if acc_stats is None:
        return stats
    else:
        acc_freq, acc_probs, acc_lse_sq = acc_stats
        sum_freq, sum_probs, sum_lse_sq = stats
        acc_freq = acc_freq + sum_freq
        acc_probs = acc_probs + sum_probs
        acc_lse_sq = acc_lse_sq + sum_lse_sq
        return acc_freq, acc_probs, acc_lse_sq


def _compute_switch_loss(acc_stats, moe_id):
    acc_freq, acc_probs, acc_lse_sq = acc_stats
    num_experts = acc_freq.size(0)
    if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
        acc_freq = all_reduce(acc_freq, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group())
    if moe_id is not None:
        update_moe_stats(moe_id, acc_freq)
    switch_loss = num_experts * torch.dot(
        F.normalize(acc_probs, p=1, dim=0), F.normalize(acc_freq.to(acc_probs.dtype), p=1, dim=0)
    )
    z_loss = acc_lse_sq / acc_freq.sum()
    loss = switch_loss + 0.1 * z_loss
    return loss.type_as(acc_lse_sq)


def _get_moe_id(obj):
    if hasattr(obj, "moe_id"):
        return obj.moe_id
    else:
        global _current_id, _stats_tracker
        obj.moe_id = _current_id
        _current_id += 1
        _stats_tracker.append(None)
        return obj.moe_id


def update_stats(module, stats):
    global _stats_tracker
    moe_id = _get_moe_id(module)
    _stats_tracker[moe_id] = _update_statistics(_stats_tracker[moe_id], stats)


def compute_total_loss():
    total_aux_loss = 0.0
    for moe_id in range(len(_stats_tracker)):
        total_aux_loss += _compute_switch_loss(_stats_tracker[moe_id], moe_id)
        _stats_tracker[moe_id] = None
    return total_aux_loss


def reset():
    global _current_id, _stats_tracker
    _current_id = 0
    _stats_tracker = []


reset()
