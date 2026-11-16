def build(native, chunk):
    import torch
    fn = native

    def run(x, offsets=None, indices=None):
        if offsets is None:
            spans = [(i, min(i + chunk, x.shape[1])) for i in range(0, x.shape[1], chunk)]
        else:
            bounds = offsets.cpu().tolist()
            spans = [(bounds[s] + c * chunk, min(bounds[s] + (c + 1) * chunk, bounds[s + 1])) for s, c in indices.cpu().tolist()]
        out = []
        for a, b in spans:
            xx = x[:, a:b].permute(0, 2, 3, 1).contiguous().reshape(-1, b - a)
            w = torch.ones(b - a, device=x.device, dtype=x.dtype)
            avg = fn(xx, w, None, 1e-05)[1].to(x.dtype).reshape(x.shape[0], *x.shape[2:])
            out.append(avg)
        return torch.stack(out, dim=1) if out else x.new_empty((x.shape[0], 0, *x.shape[2:]))
    return run
