from typing import Optional

import torch

def convert_topk_idxs(kv_head_idxs: torch.Tensor, num_q_heads: int) -> torch.LongTensor:
    """
    kv_head_idxs: (B, T, KVH) tensor of q-head indices in [0, num_q_heads-1].
    Returns head_ptrs: (B, QH, T) with head_ptrs[b, q, t] = kv_index (0..KVH-1)
    if KV head `kv_index` was assigned to q at (b,t), otherwise -1.
    """
    B, T, KVH = kv_head_idxs.shape
    QH = num_q_heads
    device = kv_head_idxs.device
    # default -1 (meaning 'no kv assigned for that q-head at that token')
    head_ptrs = torch.full((B, QH, T), -1, dtype=torch.long, device=device)
    # we want to set head_ptrs[b, q_index, t] = kv_index
    # indices for scatter must have shape (B, KVH, T): q_index per kv slot
    indices = kv_head_idxs.permute(0, 2, 1)  # (B, KVH, T)
    # values to place are kv indices 0..KVH-1, broadcasted to (B, KVH, T)
    kv_ids = torch.arange(KVH, device=device, dtype=torch.long).view(1, KVH, 1).expand(B, KVH, T)
    # scatter along dim=1 (the QH axis)
    head_ptrs.scatter_(dim=1, index=indices, src=kv_ids)
    return head_ptrs




def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              kv_head_idxs: torch.LongTensor,
              causal: bool, sm_scale: Optional[float] = 1.0) -> torch.Tensor:
    B, QH, T, D = q.shape
    KVH = k.shape[1]
    assert kv_head_idxs.shape[0] == B
    assert kv_head_idxs.shape[1] == T
    assert kv_head_idxs.shape[2] == KVH
    compute_dtype = q.dtype

    head_ptrs = torch.full((B, QH, T), fill_value=-1, device=kv_head_idxs.device)
    head_ptrs = convert_topk_idxs(kv_head_idxs, num_q_heads=QH)
    q_ = q.to(compute_dtype)

    selected_mask = head_ptrs == -1
    idx_safe = head_ptrs.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, D)  # (B, QH, T, D), long dtype
    k_ = k.gather(1, idx_safe)  # (B, QH, T, D)
    v_ = v.gather(1, idx_safe)  # (B, QH, T, D)

    # b_idxs = torch.arange(B, device=head_ptrs.device)[:, None, None]
    # t_idxs = torch.arange(T, device=head_ptrs.device)[None, None, :]
    # k__ = k[b_idxs, head_ptrs.clamp(min=0), t_idxs]
    # v__ = v[b_idxs, head_ptrs.clamp(min=0), t_idxs]
    # assert torch.allclose(k_, k__)
    # assert torch.allclose(v_, v__)

    # Compute scaled QK.
    # shape: (B, H, T, T)
    qk = torch.matmul(q_, k_.transpose(-2, -1))
    if sm_scale is not None:
        qk = qk * sm_scale

    neg_inf = float("-inf")
    qk.masked_fill_(selected_mask[:, :, None, :], neg_inf)
    # Causal mask: prevent attending to future positions
    if causal:
        # mask has True where values should be masked (upper triangular, excluding diagonal)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        # expand to (1,1,T,T)
        qk = qk.masked_fill(mask.view(1, 1, T, T), neg_inf)

    # Softmax (PyTorch's softmax is numerically stable).
    probs = torch.softmax(qk, dim=-1)
    probs.masked_fill_(probs.isnan(), 0.)
    # print(probs[0, 0, :9, :9])
    out = torch.matmul(probs, v_)
    return out.to(q.dtype)



if __name__ == '__main__':
    # small smoke test
    device = torch.device('cuda')
    B, QH, KVH, T, D = 2, 8, 2, 4096, 32
    q = torch.randn(B, QH, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, KVH, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, KVH, T, D, device=device, dtype=torch.float32)

    # choose random per-token top-2 heads
    topk = 2
    # generate random scores and pick topk heads per (b,t)
    scores = torch.rand(B, T, QH, device=device)
    _, idx = torch.topk(scores, k=topk, dim=-1)
    out = attention(q, k, v, kv_head_idxs=idx, causal=True, sm_scale=1.0 / (D ** 0.5))
    print('out shape:', out.shape)
    print('out sample:')
    print(out[0, 0, :16, :4])