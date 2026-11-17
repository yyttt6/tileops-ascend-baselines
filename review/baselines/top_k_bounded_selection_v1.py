"""Complete FP32 top-k threshold from the actual per-call k tensor.

Only the maximum requested count is transferred to the host to select the
existing Torch API's integer k argument. Every call recomputes it from live
inputs. Values, comparisons, ties, and output bits keep the original contract.
"""
import torch


def top_k_mask(logits, k):
    batch, vocab = logits.shape
    k = k.view(batch).to(logits.device, torch.long)
    if vocab == 0 or batch == 0:
        return logits.clone()
    indices = k.clamp(max=vocab) - 1
    count_dtype = torch.int32 if vocab < 2**31 else torch.int64
    # The conversion is exact: the launch count is bounded to [1, vocab].
    count = int((indices + 1).clamp(min=1).to(count_dtype).amax().item())
    values = logits.float()
    selected = values.topk(count, dim=-1, largest=True, sorted=True).values
    threshold = selected.gather(1, indices[:, None])
    keep = (values >= threshold) | (k >= vocab)[:, None]
    return logits.masked_fill(~keep, float('-inf'))
