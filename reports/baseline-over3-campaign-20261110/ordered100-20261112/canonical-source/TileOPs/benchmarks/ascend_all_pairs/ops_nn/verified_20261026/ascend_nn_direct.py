"""Direct bindings with the original ops-nn input/output alias contract."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, _add_layout, source_hashes
from .ascend_graph_call import GraphCall


def opsnn_direct_rms(case):
    import torch
    from benchmarks.api import Implementation

    binding = GraphCall("ops-nn")

    def run(x, weight):
        output = torch.empty_like(x)
        rstd = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
        binding.call("RmsNorm", [x.contiguous(), weight.contiguous()], [output, rstd],
                     {"epsilon": float(case.params["eps"])})
        return output

    files = [Path(__file__), Path(__file__).with_name("ascend_graph_call.py")]
    for root in (binding.vendor, LIBRARIES / "ops-nn/norm/rms_norm"):
        files.extend(p for p in root.rglob("*") if p.is_file() and
                     p.suffix in (".cpp", ".h", ".py", ".json", ".so", ".o"))
    return Candidate(
        Implementation(run), "ops-nn::RmsNorm via CANN graph JIT",
        "Original standalone weighted RMS, unchanged BF16 inputs and manifest epsilon.",
        source_hashes(*files),
        limitations=("Own source compilation and matching launch must be proven.",),
        cleanup=binding.cleanup,
    )


def opsnn_inplace_add(case):
    import torch
    from benchmarks.api import Implementation

    _, _, shape, rows, hidden, pack = _add_layout(case)
    binding = GraphCall("ops-nn")
    if not (binding.vendor / "op_impl/ai_core/tbe/classicpilot_nn_impl/dynamic/"
            "inplace_add_rms_norm_apt.py").is_file():
        raise UnsupportedCase("The original InplaceAddRmsNorm must be packaged first.")

    def run(x, y):
        first = pack(x).clone(memory_format=torch.contiguous_format)
        second = pack(y).clone(memory_format=torch.contiguous_format)
        gamma = torch.ones(hidden, device=x.device, dtype=x.dtype)
        rstd = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
        binding.call("InplaceAddRmsNorm", [first, second, gamma],
                     [first, rstd, second], {"epsilon": 1e-6})
        return second.view(shape)

    files = [Path(__file__), Path(__file__).with_name("ascend_graph_call.py")]
    for root in (binding.vendor, LIBRARIES / "ops-nn/norm/inplace_add_rms_norm",
                 LIBRARIES / "ops-nn/norm/add_rms_norm"):
        files.extend(p for p in root.rglob("*") if p.is_file() and
                     p.suffix in (".cpp", ".h", ".py", ".json", ".so", ".o"))
    return Candidate(
        Implementation(run), "ops-nn::InplaceAddRmsNorm residual output via CANN graph JIT",
        "Per-call broadcast materialization and private 2D copies; original x2 alias contains residual sum; restore output shape.",
        source_hashes(*files),
        limitations=("Own source compilation and matching launch must be proven.",),
        cleanup=binding.cleanup,
    )
