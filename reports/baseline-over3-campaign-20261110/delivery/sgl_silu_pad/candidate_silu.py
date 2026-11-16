import torch
import torch.nn.functional as F
def build(case,fn):
    x, = case.inputs
    size = x.numel() // 2
    tail = -size % 1024
    counts = torch.tensor([(size + tail) // 1024], device=x.device, dtype=torch.int32)
    def run(x):
        gate, up = x.chunk(2, dim=-1)
        gate = gate.contiguous().view(-1)
        gate = (F.pad(gate, (0, tail)) if tail else gate).view(-1, 1024)
        up = up.contiguous().view(-1)
        up = (F.pad(up, (0, tail)) if tail else up).view(-1, 1024)
        out, _ = fn(torch.cat((gate, up), dim=-1), counts, 1, need_quant=False)
        return out.reshape(-1)[:size].reshape(*x.shape[:-1], x.shape[-1] // 2)
    return run
