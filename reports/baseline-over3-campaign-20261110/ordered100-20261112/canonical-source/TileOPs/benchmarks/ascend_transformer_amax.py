"""Select the original FlashAttention API's QK maximum using a basis query."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_math_more import _MathCall


def transformer_amax(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    if case.inputs[0].ndim != 2 or case.params["dim"] != -1:
        raise UnsupportedCase("This pilot reduces independent packed rows.")
    binding = _MathCall(
        "aclnnFlashAttentionScore",
        [C.c_void_p] * 8 + [C.c_double] * 2 + [C.c_int64] * 3 + [C.c_char_p]
        + [C.c_int64] * 2 + [C.c_void_p] * 4, library="ops-transformer",
    )

    def run(x):
        rows, width = x.shape
        # q=[1,0,...] and k=[x_i,0,...] give QK=x exactly.
        keys = F.pad(x.reshape(rows, width, 1, 1), (0, 127)).contiguous()
        query = torch.zeros((rows, 1, 1, 128), dtype=x.dtype, device=x.device)
        query[..., 0] = 1
        maximum = torch.empty((rows, 1, 1, 8), dtype=torch.float32, device=x.device)
        total = torch.empty_like(maximum)
        out = torch.empty_like(query)
        record = binding.record()
        ptr = lambda t: binding.tensor(t, record)
        kptr = ptr(keys)
        binding.call([ptr(query), kptr, kptr, None, None, None, None, None,
                      1.0, 1.0, 2147483647, 2147483647, 1, b"BSND", 0, 0,
                      ptr(maximum), ptr(total), None, ptr(out)], record, x.device)
        result = maximum[:, 0, 0, 0].to(x.dtype)
        return result[:, None] if case.params["keepdim"] else result

    return Candidate(
        Implementation(run), "ops-transformer::aclnnFlashAttentionScore.softmaxMax",
        "Native exposed maximum; basis query, input padding and scale1 make QK equal each input row. "
        "Original attention kernel unchanged; discard unused outputs.",
        source_hashes(Path(__file__)) | binding.hashes(
            LIBRARIES / "ops-transformer/attention/flash_attention_score"),
        limitations=("Large temporary padded keys and full native attention work are included in adapter cost.",
                     "Own device-kernel attribution and numerical check required."),
        cleanup=binding.cleanup,
    )
