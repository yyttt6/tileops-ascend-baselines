"""Memory-bounded, device-resident PyTorch for the large MoE and SSD cases.

No values are copied to the host by either candidate. Dynamic ``nonzero`` in
MoE can synchronize to obtain its result shape, but routing and arithmetic stay
on the input device. Inputs are never mutated. The CPU SSD oracle deliberately
uses the original elementwise outer-product reduction, bounded by chunk/head
tiles, independently of the candidate's batched matrix multiplication.
"""

import torch
import torch.nn.functional as F

OPS = {"FusedMoESharedExpertFwdOp", "SSDChunkStateFwdOp"}
MUTATED = {}


def _routing(logits, bias, params):
    logits = logits.float()
    scores = logits.softmax(-1) if params["scoring_func"] == "softmax" else logits.sigmoid()
    selection = scores if bias is None else scores + bias
    groups = params["num_expert_group"]
    if groups > 1:
        grouped = selection.reshape(selection.shape[0], groups, -1)
        if bias is None:
            group_scores = grouped.amax(-1)
        else:
            group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        kept = group_scores.argsort(dim=-1, descending=True, stable=True)
        kept = kept[:, :params["topk_group"]]
        mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, kept, True)
        selection = selection.masked_fill(
            ~mask.repeat_interleave(selection.shape[1] // groups, dim=1), -float("inf")
        )
    ids = selection.topk(params["top_k"], dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if params["renormalize"]:
        weights = weights / weights.sum(-1, keepdim=True)
    return ids, weights


def _activation(gate, up, name):
    if name == "silu_and_mul":
        return F.silu(gate) * up
    if name == "gelu_and_mul":
        return F.gelu(gate) * up
    raise ValueError(f"unsupported activation: {name}")


def _moe(params, hidden, logits, w_gate_up, w_down, bias=None,
         shared_w_gate_up=None, shared_w_down=None):
    ids, weights = _routing(logits, bias, params)
    routed = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    # Never convert the full E x 2F x H expert bank to FP32.
    for expert in range(w_gate_up.shape[0]):
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        gate_up_weight = w_gate_up[expert].float()
        down_weight = w_down[expert].float()
        for start in range(0, tokens.numel(), 512):
            rows = tokens[start:start + 512]
            choices = slots[start:start + 512]
            gate, up = (hidden[rows].float() @ gate_up_weight.T).chunk(2, -1)
            values = _activation(gate, up, params["activation"]) @ down_weight.T
            routed.index_add_(0, rows, values * weights[rows, choices, None])
        del gate_up_weight, down_weight
    routed = (routed * params["routed_scaling_factor"]).to(hidden.dtype)
    if shared_w_gate_up is None:
        return None, routed

    width = shared_w_down.shape[1]
    shard = width // params["tp_size"]
    lo = params["tp_rank"] * shard
    hi = lo + shard
    shared = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    # Split the shared intermediate axis, while using the complete H reduction.
    for start in range(lo, hi, 2048):
        end = min(start + 2048, hi)
        gate_weight = shared_w_gate_up[start:end].float()
        up_weight = shared_w_gate_up[width + start:width + end].float()
        down_weight = shared_w_down[:, start:end].float()
        for row in range(0, hidden.shape[0], 512):
            x = hidden[row:row + 512].float()
            gate, up = x @ gate_weight.T, x @ up_weight.T
            shared[row:row + 512].add_((F.silu(gate) * up) @ down_weight.T)
        del gate_weight, up_weight, down_weight
    return shared.to(hidden.dtype), routed


def _ssd(x, bmat, dt, dacs, seq_idx=None):
    batch, length, heads, d_head = x.shape
    chunks, q = dt.shape[-2:]
    groups, d_state = bmat.shape[-2:]
    per_group = heads // groups
    count = batch * chunks
    xx = x.reshape(count, q, heads, d_head)
    bb = bmat.reshape(count, q, groups, d_state)
    ddt = dt.permute(0, 2, 1, 3).reshape(count, heads, q)
    da = dacs.permute(0, 2, 1, 3).reshape(count, heads, q)
    seq = None if seq_idx is None else seq_idx.reshape(count, q)
    out = torch.empty((count, heads, d_head, d_state), device=x.device, dtype=torch.float32)
    for start in range(0, count, 16):
        stop = min(start + 16, count)
        for group in range(groups):
            for head in range(group * per_group, (group + 1) * per_group, 8):
                end = min(head + 8, (group + 1) * per_group)
                decay = da[start:stop, head:end].float()
                weight = (decay[..., -1:] - decay).clamp(max=0).exp()
                weight = weight * ddt[start:stop, head:end].float()
                if seq is not None:
                    s = seq[start:stop]
                    weight = weight * ((s[:, -1:] >= 0) & (s == s[:, -1:]))[:, None]
                # Match reference's w*B before contraction, avoiding Q x N x P.
                weighted_b = weight[..., None] * bb[start:stop, :, group].float()[:, None]
                tokens = xx[start:stop, :, head:end].float().permute(0, 2, 3, 1)
                out[start:stop, head:end] = tokens @ weighted_b
    return out.reshape(batch, chunks, heads, d_head, d_state)


def bind(case):
    """Bind scalar metadata without materializing ``case.arguments``."""
    op = case._call.signature.name
    if op == "FusedMoESharedExpertFwdOp":
        params = dict(case._call.params)
        return lambda *inputs: _moe(params, *inputs)
    if op == "SSDChunkStateFwdOp":
        return _ssd
    raise KeyError(op)


def cpu_reference(case, inputs):
    """Independent full-output CPU oracle, with a bounded SSD outer product."""
    if any(t is not None and t.device.type != "cpu" for t in inputs):
        raise ValueError("cpu_reference requires captured CPU inputs")
    if case._call.signature.name == "FusedMoESharedExpertFwdOp":
        # Current workload already computes one expert at a time; it does not
        # share candidate routing/MLP helpers or candidate's shared-axis tiling.
        return case.reference(*inputs)
    if case._call.signature.name != "SSDChunkStateFwdOp":
        raise KeyError(case._call.signature.name)
    x, bmat, dt, dacs, seq_idx = inputs
    batch, length, heads, d_head = x.shape
    chunks, q = dt.shape[-2:]
    groups, n = bmat.shape[-2:]
    per_group = heads // groups
    xx = x.reshape(batch, chunks, q, heads, d_head)
    bb = bmat.reshape(batch, chunks, q, groups, n)
    seq = None if seq_idx is None else seq_idx.reshape(batch, chunks, q)
    out = torch.empty((batch, chunks, heads, d_head, n), dtype=torch.float32)
    for b in range(batch):
        for c in range(chunks):
            for h in range(0, heads, 8):
                end = min(h + 8, heads)
                decay = dacs[b, h:end, c].float()
                weight = torch.exp(torch.clamp(decay[:, -1:] - decay, max=0.0))
                weight = weight * dt[b, h:end, c].float()
                if seq is not None:
                    s = seq[b, c]
                    weight *= ((s[-1] >= 0) & (s == s[-1]))[None]
                group_ids = torch.arange(h, end) // per_group
                b_heads = bb[b, c, :, group_ids].float()
                weighted_b = weight.T[..., None] * b_heads
                contrib = weighted_b[..., None] * xx[b, c, :, h:end].float().unsqueeze(-2)
                out[b, c, h:end] = contrib.sum(0).transpose(-1, -2)
    return out
