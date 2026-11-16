"""Original ops-nn graph APIs with per-call manifest layout adaptation."""
from pathlib import Path


def adaptive_max_pool(case):
    import torch
    from .graph import GraphCall
    from benchmarks.ascend_nn_graph import _candidate

    binding = GraphCall("ops-nn")
    indices_requested = "Indices" in case._call.signature.name
    size = case.params["output_size"]

    def run(x):
        target = (size, size) if isinstance(size, int) or size is None else size
        target = tuple(x.shape[-2 + i] if v is None else v for i, v in enumerate(target))
        out = torch.empty((*x.shape[:-2], *target), device=x.device, dtype=x.dtype)
        indices = torch.empty_like(out, dtype=torch.int64)
        binding.call("AdaptiveMaxPool2d", [x.contiguous()], [out, indices],
                     {"output_size": list(target), "argmax_dtype": 9})
        return (out, indices) if indices_requested else out

    return _candidate(binding, run, "AdaptiveMaxPool2d",
                      "Original adaptive windows and flattened spatial indices; exact manifest output_size; per-call contiguous input.",
                      "pooling/adaptive_max_pool2d", extra_sources=(Path(__file__),Path(__file__).with_name("graph.py")))
