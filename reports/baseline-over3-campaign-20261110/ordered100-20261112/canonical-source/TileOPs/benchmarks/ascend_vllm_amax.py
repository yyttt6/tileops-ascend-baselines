"""A basis-query use of the original exposed native softmax maximum."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes, vllm_native


def vllm_amax(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    if case.inputs[0].ndim != 2 or case.params["dim"] != -1:
        raise UnsupportedCase("This pilot reduces independent packed rows.")
    ops, binary, vendor = vllm_native()
    fn = ops.npu_sparse_flash_attention

    def run(x):
        rows, width = x.shape
        keys = F.pad(x.reshape(rows, width, 1, 1), (0, 511)).contiguous()
        query = torch.zeros((rows, 1, 1, 512), device=x.device, dtype=x.dtype)
        query[..., 0] = 1
        qr = torch.zeros((rows, 1, 1, 64), device=x.device, dtype=x.dtype)
        kr = torch.zeros((rows, width, 1, 64), device=x.device, dtype=x.dtype)
        indices = torch.arange(width, device=x.device, dtype=torch.int32)
        indices = indices.reshape(1, 1, 1, width).expand(rows, 1, 1, width).contiguous()
        _, maximum, _ = fn(query, keys, keys, indices, 1.0, query_rope=qr, key_rope=kr,
                           sparse_block_size=1, sparse_mode=0, attention_mode=2,
                           layout_query="BSND", layout_kv="BSND", return_softmax_lse=True)
        result = maximum.reshape(rows).to(x.dtype)
        return result[:, None] if case.params["keepdim"] else result

    root = LIBRARIES / "vllm-ascend/csrc/attention/sparse_flash_attention"
    sources = [p for p in root.rglob("*") if p.suffix in (".h", ".cpp", ".json")]
    deployed = [p for p in vendor.rglob("*") if p.is_file() and p.suffix in (".so", ".o", ".json")]
    return Candidate(
        Implementation(run), "vllm-ascend::npu_sparse_flash_attention.softmax_max",
        "Basis Q, zero-padded K=x and zero rope make native QK=x; all indices, scale1, select exposed maximum.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_baselines.py"),
                      binary, *sources, *deployed),
        limitations=("Padded keys require about8GiB and all native attention work counts toward cost.",
                     "Own native kernel attribution and actual numerical validation are required."),
    )
