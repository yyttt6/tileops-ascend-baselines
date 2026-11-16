"""Hierarchical original top-k; merge original logits, never block probabilities."""
from pathlib import Path


def topk_select(case):
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from .hashing import source_hashes
    binding = GraphCall('ops-transformer')
    topk = case.params['topk']

    def select(values, valid, count):
        count = min(count, values.shape[1])
        chosen = torch.zeros((values.shape[0], count), dtype=torch.int64, device=values.device)
        live = valid.any(1).nonzero().flatten()
        if live.numel():
            logits = values.index_select(0, live).contiguous()
            out = torch.empty((live.numel(), count), dtype=torch.float32, device=values.device)
            ids = torch.empty_like(out, dtype=torch.int32)
            rows = torch.empty_like(ids)
            binding.call('MoeGatingTopKSoftmax', [logits, None], [out, ids, rows], {'k': count})
            chosen.index_copy_(0, live, ids.long())
        return chosen

    def global_selection(values, valid, count):
        indices = torch.arange(values.shape[1], device=values.device)[None].expand(values.shape[0], -1)
        while values.shape[1] > 2048:
            picked_values, picked_valid, picked_indices = [], [], []
            for start in range(0, values.shape[1], 2048):
                block = values[:, start:start+2048]
                flags = valid[:, start:start+2048]
                chosen = select(block, flags, count)
                picked_values.append(block.gather(1, chosen))
                picked_valid.append(flags.gather(1, chosen))
                picked_indices.append(indices[:, start:start+2048].gather(1, chosen))
            values, valid, indices = [torch.cat(parts, 1) for parts in (picked_values, picked_valid, picked_indices)]
        chosen = select(values, valid, count)
        return indices.gather(1, chosen)

    def run(scores, starts, ends):
        b, s, n, g = scores.shape
        rows = scores.movedim(2, -1).contiguous().reshape(-1, n)
        low = starts[:, :, None].expand(b, s, g).reshape(-1)
        high = ends[:, :, None].expand(b, s, g).reshape(-1)
        results = []
        for begin in range(0, rows.shape[0], 128):
            end = min(begin+128, rows.shape[0])
            positions = torch.arange(n, device=scores.device)[None]
            valid = (positions >= low[begin:end, None]) & (positions < high[begin:end, None])
            active = rows[begin:end].masked_fill(~valid, -float('inf'))
            selected = []
            for k0 in range(0, topk, 1024):
                count = min(1024, topk-k0)
                ids = global_selection(active, valid, count)
                inside = valid.gather(1, ids)
                selected.append(torch.where(inside, ids, n).int())
                valid.scatter_(1, ids, False)
                active.scatter_(1, ids, -float('inf'))
            results.append(torch.cat(selected, -1))
        return torch.cat(results).reshape(b, s, g, topk)

    result = _candidate(binding, run, 'MoeGatingTopKSoftmax', 'moe/moe_gating_top_k_softmax',
                        'Window masks and sentinel padding from integer metadata; original native top-k on <=2048 experts/row and <=1024 selected entries. Hierarchical candidates gather original FP32 logits for every merge, preserving cross-block ordering. No external sort/top-k or tolerance change; full original selected-value/uniqueness checks remain required.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
