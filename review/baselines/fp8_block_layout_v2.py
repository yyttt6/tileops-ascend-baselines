"""Historical complete FP8 block layout alternative, unchanged arithmetic."""
import torch
import torch.nn.functional as F

def fp8_block(w):
    n, k = w.shape
    wf = w.float()
    pn = -n % 128
    pk = -k % 128
    padded = F.pad(wf, (0, pk, 0, pn)) if pn or pk else wf
    tiles = padded.reshape((n + pn) // 128, 128, (k + pk) // 128, 128)
    am = tiles.abs().amax((1, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = torch.where(am > 0, am / fp8_max, torch.ones_like(am))
    q = (tiles / scale[:, None, :, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (q.reshape(n + pn, k + pk)[:n, :k], scale)
