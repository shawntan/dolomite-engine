"""
Fused MoKV Attention
===============

E: number of q heads / experts
k: number of kv heads / top-k
D: head dim

Inputs: 
- Top-k indices: (batch, length, k)
- Q:    (batch, E, length, D)
- K, V: (batch, k, length, D)

Outputs:
- O:    (batch, E, length, D)

"""

import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()
ALLOW_TF32 = tl.constexpr(True)
NEG_INF = tl.constexpr(-1.0e6)

def convert_topk_idxs(kv_idxs: torch.Tensor, num_q_heads: int) -> torch.LongTensor:
    """
    kv_head_idxs: (B, T, KVH) tensor of q-head indices in [0, num_q_heads-1].
    Returns head_ptrs: (B, QH, T) with head_ptrs[b, q, t] = kv_index (0..KVH-1)
    if KV head `kv_index` was assigned to q at (b,t), otherwise -1.
    """
    B, T, KVH = kv_idxs.shape
    QH = num_q_heads
    device = kv_idxs.device
    # default -1 (meaning 'no kv assigned for that q-head at that token')
    head_ptrs = torch.full((B, QH, T), -1, dtype=torch.long, device=device)
    # we want to set head_ptrs[b, q_index, t] = kv_index
    # indices for scatter must have shape (B, KVH, T): q_index per kv slot
    indices = kv_idxs.permute(0, 2, 1)  # (B, KVH, T)
    # values to place are kv indices 0..KVH-1, broadcasted to (B, KVH, T)
    kv_ids = torch.arange(KVH, device=device, dtype=torch.long).view(1, KVH, 1).expand(B, KVH, T)
    # scatter along dim=1 (the QH axis)
    head_ptrs.scatter_(dim=1, index=indices, src=kv_ids)
    return head_ptrs


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, 
                    q,
                    K_b_ptr, k_stride: tl.constexpr,
                    V_b_ptr, v_stride: tl.constexpr,
                    KV_idx_bh_ptr, kvi_stride: tl.constexpr,
                    dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    head_dim_idxs = tl.arange(0, HEAD_DIM)
    same_strides = k_stride == v_stride


    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        off_n_idxs = start_n + offs_n

        off_kvh_idxs = tl.load(KV_idx_bh_ptr + kvi_stride[2] * off_n_idxs)
        kv_mask = off_kvh_idxs != -1
        off_kvh_idxs = tl.where(kv_mask, off_kvh_idxs, 0)

        if same_strides:
            kv_idxs = (k_stride[1] * off_kvh_idxs + k_stride[2] * off_n_idxs)[:, None] + k_stride[3] * head_dim_idxs[None, :]
            k = tl.load(K_b_ptr + kv_idxs).T
            v = tl.load(V_b_ptr + kv_idxs)
        else:
            k = tl.load(K_b_ptr + (k_stride[1] * off_kvh_idxs + k_stride[2] * off_n_idxs)[:, None] + k_stride[3] * head_dim_idxs[None, :]).T
            v = tl.load(V_b_ptr + (v_stride[1] * off_kvh_idxs + v_stride[2] * off_n_idxs)[:, None] + v_stride[3] * head_dim_idxs[None, :])

        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * qk_scale

        if STAGE == 2: # on band
            mask = (offs_m[:, None] >= off_n_idxs[None, :]) & kv_mask[None, :]
            qk =  tl.where(mask, qk, NEG_INF)
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            p = tl.where(mask, p, 0.)
        else: # off band
            qk =  tl.where(kv_mask[None, :], qk, NEG_INF)
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            p = tl.where(kv_mask[None, :], p, 0.)

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        # -- update output accumulator --
        # acc = acc * alpha[:, None]
        acc *= alpha[:, None]
        # prepare p and v for the dot
        p = p.to(dtype)
        acc = tl.dot(p, v, acc, allow_tf32=ALLOW_TF32)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    # if not isinstance(nargs["q"], TensorDescriptor):
    #     return
    # nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    # if nargs["FP8_OUTPUT"]:
    #     nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    # else:
    #     nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    # nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    # nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.autotune(configs=list(filter(keep, configs)),
                 key=["q_stride", "k_stride", "v_stride", "N_CTX", "HEAD_DIM", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,
              batch_size: tl.constexpr,
              QH: tl.constexpr, KVH: tl.constexpr,
              Q_ptr, q_stride: tl.constexpr,
              K_ptr, k_stride: tl.constexpr,
              V_ptr, v_stride: tl.constexpr,
              KV_idx_ptr, kvi_stride: tl.constexpr,
              O_ptr, o_stride: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              FP8_OUTPUT: tl.constexpr,
              STAGE: tl.constexpr,
              warp_specialize: tl.constexpr,
              IS_HOPPER: tl.constexpr):
    dtype = Q_ptr.dtype.element_ty
    # tl.static_print(BLOCK_N, HEAD_DIM)
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // QH
    off_qh = off_bh % QH
    off_kvh = off_qh

    y_dim = batch_size * QH * N_CTX
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HEAD_DIM)

    K_b_ptr = K_ptr + k_stride[0] * off_b
    V_b_ptr = V_ptr + v_stride[0] * off_b
    KV_idx_bh_ptr = KV_idx_ptr + kvi_stride[0] * off_b + kvi_stride[1] * off_kvh
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    # q = Q_ptr.load([qo_offset_y, 0])
    q = tl.load(
        Q_ptr +
        q_stride[0] * off_b +
        q_stride[1] * off_qh +
        q_stride[2] * offs_m[:, None] + 
        q_stride[3] * offs_h[None, :]
    )
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i,
            q,
            K_b_ptr, k_stride,
            V_b_ptr, v_stride,
            KV_idx_bh_ptr, kvi_stride,
            dtype, start_m, qk_scale,  #
            BLOCK_M, HEAD_DIM, BLOCK_N,  #
            4 - STAGE, offs_m, offs_n, N_CTX,  #
            warp_specialize, IS_HOPPER
        )
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i,
            q,
            K_b_ptr, k_stride,
            V_b_ptr, v_stride,
            KV_idx_bh_ptr, kvi_stride,
            dtype, start_m, qk_scale,  #
            BLOCK_M, HEAD_DIM, BLOCK_N,  #
            2, offs_m, offs_n, N_CTX,  #
            warp_specialize, IS_HOPPER
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    acc = tl.where((l_i == 0)[:, None], 0., acc)
    m_ptrs = M + off_bh * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(
        O_ptr +
        o_stride[0] * off_b +
        o_stride[1] * off_qh +
        o_stride[2] * offs_m[:, None] + 
        o_stride[3] * offs_h[None, :],
        acc.to(dtype)
    )


@triton.jit
def _attn_bwd_preprocess(O, o_stride: tl.constexpr,
                         DO, do_stride: tl.constexpr,
                         Delta,  #
                         batch_size, num_heads, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    batch_id = off_hz // num_heads
    head_id = off_hz % num_heads
    head_idxs = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + batch_id * o_stride[0] + head_id * o_stride[1] + off_m[:, None] * o_stride[2] + head_idxs[None, :] * o_stride[3])
    do = tl.load(DO + batch_id * do_stride[0] + head_id * do_stride[1] + off_m[:, None] * do_stride[2] + head_idxs[None, :] * do_stride[3]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv, k, v, sm_scale,
                   kv_mask,
                   Q_ptr, q_stride,
                   DO_ptr, do_stride,
                   M, D,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):

    dtype = Q_ptr.dtype.element_ty
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    QT_blk_ptrs = Q_ptr + offs_m[None, :] * q_stride[2] + offs_k[:, None] * q_stride[3]
    DO_blk_ptrs = DO_ptr + offs_m[:, None] * do_stride[2] + offs_k[None, :] * do_stride[3]
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in tl.range(num_steps):
        qT = tl.load(QT_blk_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT, allow_tf32=ALLOW_TF32)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        # if MASK:
        #     mask = (offs_m[None, :] >= offs_n[:, None])
        #     pT = tl.where(mask, pT, 0.0)
        mask = (offs_m[None, :] >= offs_n[:, None]) & kv_mask[:, None]
        pT = tl.where(mask, pT, 0.0)

        do = tl.load(DO_blk_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(dtype)
        dv += tl.dot(ppT, do, allow_tf32=ALLOW_TF32)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(dtype)
        dk += tl.dot(dsT, tl.trans(qT), allow_tf32=ALLOW_TF32)
        # Increment pointers.
        curr_m += step_m
        QT_blk_ptrs += step_m * q_stride[2]
        DO_blk_ptrs += step_m * do_stride[2]
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q,
                 K_b_ptr, k_stride,
                 V_b_ptr, v_stride,
                 KVI_bh_ptr, kvi_stride,
                 do, m, D,
                 # shared by Q/K/V/DO.
                 H, N_CTX,  #
                 head_id,
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):

    dtype = K_b_ptr.dtype.element_ty
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K_b_ptr + offs_n[None, :] * k_stride[2] + offs_k[:, None] * k_stride[3]
    vT_ptrs = V_b_ptr + offs_n[None, :] * v_stride[2] + offs_k[:, None] * v_stride[3] 
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in tl.range(num_steps):
        # head_ids = tl.zeros_like(offs_n) + head_id
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        off_kvh_idxs = tl.load(KVI_bh_ptr + kvi_stride[2] * offs_n)
        kv_mask = off_kvh_idxs != -1
        kT = tl.load(kT_ptrs + off_kvh_idxs[None, :] * k_stride[1], mask=kv_mask[None, :])
        vT = tl.load(vT_ptrs + off_kvh_idxs[None, :] * v_stride[1], mask=kv_mask[None, :])
        qk = tl.dot(q, kT, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        # if MASK:
        #     offs_n = curr_n + tl.arange(0, BLOCK_N2)
        #     mask = (offs_m[:, None] >= offs_n[None, :])
        #     p = tl.where(mask, p, 0.0)
        mask = (offs_m[:, None] >= offs_n[None, :]) & kv_mask[None, :]
        p = tl.where(mask, p, 0.0)

        # Compute dP and dS.
        dp = tl.dot(do, vT, allow_tf32=ALLOW_TF32).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT), allow_tf32=ALLOW_TF32)
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * k_stride[2]
        vT_ptrs += step_n * v_stride[2]
    return dq


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M1': bm1, 'BLOCK_N1': bn1, 'BLOCK_M2': bm2, 'BLOCK_N2': bn2}, num_stages=s, num_warps=w)
#         for bm1 in [32, 64] \
#         for bn1 in [64] \
#         for bm2 in [64]\
#         for bn2 in [32, 64] \
#         for w in [4, 8] \
#         for s in [2, 5] \
#         if not (bm2 > 64 or bn1 > 64) or w == 8
#     ],
#     key=["HEAD_DIM", "N_CTX"],
# )
@triton.jit
def _attn_bwd(Q_ptr, q_stride: tl.constexpr,
              K_ptr, k_stride: tl.constexpr,
              V_ptr, v_stride: tl.constexpr,
              sm_scale,
              DO_ptr, do_stride: tl.constexpr,
              DQ_ptr, dq_stride: tl.constexpr,
              DK_ptr, dk_stride: tl.constexpr,
              DV_ptr, dv_stride: tl.constexpr,
              KVI_ptr, kvi_stride: tl.constexpr,
              M, D,
              # shared by Q/K/V/DO.
              H, N_CTX,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              CAUSAL: tl.constexpr,
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    batch_id = bhid // H
    head_id = bhid % H
    # adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    start_n = pid * BLOCK_N1
    start_m = 0
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # offset pointers for batch/head
    # Q_bh_ptr = adj
    # K_bh_ptr = adj
    # V_bh_ptr = adj
    # DO_bh_ptr = adj
    # DQ_bh_ptr = adj
    # DK_bh_ptr = adj
    # DV_bh_ptr = adj
    Q_bh_ptr = Q_ptr + q_stride[0] * batch_id + q_stride[1] * head_id


    DO_bh_ptr = DO_ptr + do_stride[0] * batch_id + do_stride[1] * head_id
    DQ_bh_ptr = DQ_ptr + dq_stride[0] * batch_id + dq_stride[1] * head_id

    # head_ids = tl.zeros_like(offs_n) + head_id
    off_kvh_idxs = tl.load(KVI_ptr + kvi_stride[0] * batch_id + kvi_stride[1] * head_id + kvi_stride[2] * offs_n)
    kv_mask = off_kvh_idxs != -1
    K_bh_ptr = K_ptr + k_stride[0] * batch_id + k_stride[1] * off_kvh_idxs[:, None]
    V_bh_ptr = V_ptr + v_stride[0] * batch_id + v_stride[1] * off_kvh_idxs[:, None]
    DK_bh_ptr = DK_ptr + dk_stride[0] * batch_id + dk_stride[1] * off_kvh_idxs[:, None]
    DV_bh_ptr = DV_ptr + dv_stride[0] * batch_id + dv_stride[1] * off_kvh_idxs[:, None]

    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    # k = tl.load(K_bh_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    # v = tl.load(V_bh_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    k = tl.load(K_bh_ptr + (offs_n[:, None] * k_stride[2] + offs_k[None, :] * k_stride[3]), mask=kv_mask[:, None])
    v = tl.load(V_bh_ptr + (offs_n[:, None] * v_stride[2] + offs_k[None, :] * v_stride[3]), mask=kv_mask[:, None])

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv, k, v, sm_scale, kv_mask,
            Q_bh_ptr, q_stride,
            DO_bh_ptr, do_stride,
            M, D,
            H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps,
            MASK=True,
        )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv, k, v, sm_scale, kv_mask,
        Q_bh_ptr, q_stride,
        DO_bh_ptr, do_stride,
        M, D,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV_bh_ptr + (offs_n[:, None] * dv_stride[2] + offs_k[None, :] * dv_stride[3])
    dk_ptrs = DK_bh_ptr + (offs_n[:, None] * dk_stride[2] + offs_k[None, :] * dk_stride[3])

    tl.store(dv_ptrs, dv, mask=kv_mask[:, None])
    # Write back dK.
    dk *= sm_scale
    tl.store(dk_ptrs, dk, mask=kv_mask[:, None])

    # ---------------------------------------------------------------------------------------
    # THIS BLOCK DOES DQ:

    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q_bh_ptr + offs_m[:, None] * dq_stride[2] + offs_k[None, :] * dq_stride[3])
    do = tl.load(DO_bh_ptr + offs_m[:, None] * do_stride[2] + offs_k[None, :] * do_stride[3])

    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q,
            K_ptr + batch_id * k_stride[0], k_stride,
            V_ptr + batch_id * v_stride[0], v_stride,
            KVI_ptr + batch_id * kvi_stride[0] + head_id * kvi_stride[1], kvi_stride,
            do, m, D,
            H, N_CTX,
            head_id,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
            MASK=True,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq, q,
        K_ptr + batch_id * k_stride[0], k_stride,
        V_ptr + batch_id * v_stride[0], v_stride,
        KVI_ptr + batch_id * kvi_stride[0] + head_id * kvi_stride[1], kvi_stride,
        do, m, D,
        H, N_CTX,
        head_id,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps,
        MASK=False,
    )
    # Write back dQ.
    dq_ptrs = DQ_bh_ptr + offs_m[:, None] * dq_stride[2] + offs_k[None, :] * dq_stride[3]
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, kv_idxs, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        num_q_heads = q.shape[1]
        kv_head_idxs = convert_topk_idxs(kv_idxs, num_q_heads=num_q_heads)
        # print(kv_head_idxs[0][:, :10].t())
        batch_size, num_heads, length, head_dim = k.size()
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        _attn_fwd[grid](
            sm_scale, M,
            q.shape[0],
            q.shape[1], k.shape[1],
            q, q.stride(),
            k, k.stride(),
            v, v.stride(),
            kv_head_idxs, kv_head_idxs.stride(),
            o, o.stride(),
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=is_hopper(),
            **extra_kern_args
        )
        ctx.save_for_backward(q, k, v, o, M, kv_head_idxs)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, M, kv_head_idxs = ctx.saved_tensors
        # assert do.is_contiguous()
        do = do.contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 8, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32 # to tune warps and everything.
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M) # batch_size, num_heads, length
        _attn_bwd_preprocess[pre_grid](
            o, o.stride(),
            do, do.stride(),
            delta,
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, q.stride(),
            arg_k, arg_k.stride(),
            v, v.stride(),
            ctx.sm_scale,
            do, do.stride(),
            dq, dq.stride(),
            dk, dk.stride(),
            dv, dv.stride(),
            kv_head_idxs, kv_head_idxs.stride(),
            M, delta,
            N_HEAD, N_CTX,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dq, dk, dv, None, None, None, None, None


def attention(q, k, v, kv_idxs, causal, sm_scale, warp_specialize=True):
    return _attention.apply(q, k, v, kv_idxs, causal, sm_scale, warp_specialize)


TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')




