"""Sort exactly represented input values without an unnecessary FP32 buffer.

FP16/BF16 values embed exactly into FP32, so their ordering, ties, and comparisons
are unchanged. Other dtypes retain the original FP32 conversion. This is a full
Torch API composition; no device kernel or precision setting is modified.
"""
import torch


def top_k_mask(logits, k):
    batch, vocab = logits.shape
    k = k.view(batch).to(logits.device, torch.long)
    if vocab == 0:
        return logits.clone()
    values = logits if logits.dtype in (torch.float16, torch.bfloat16, torch.float32) else logits.float()
    kth = values.sort(-1, descending=True).values.gather(1, (k.clamp(max=vocab) - 1)[:, None])
    kept = (values >= kth) | (k >= vocab)[:, None]
    return logits.masked_fill(~kept, float('-inf'))
