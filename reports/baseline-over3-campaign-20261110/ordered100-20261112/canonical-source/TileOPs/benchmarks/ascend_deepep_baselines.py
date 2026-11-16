"""One-rank use of DeepEP's original expanding dispatch API."""

from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes


def deepep_prepermute(case):
    from datetime import timedelta
    import importlib
    import sys
    import tempfile
    import torch
    import torch.distributed as dist
    from benchmarks.api import Implementation

    root = LIBRARIES / "DeepEP-Ascend"
    build = root / f"build/lib.linux-x86_64-cpython-{sys.version_info.major}{sys.version_info.minor}"
    # Keep the original Python API and the extension from the same local build.
    for path in (root / "deep_ep").rglob("*.py"):
        built = build / "deep_ep" / path.relative_to(root / "deep_ep")
        if not built.exists() or built.read_bytes() != path.read_bytes():
            raise RuntimeError(f"DeepEP build contains stale Python source: {path}")
    sys.path.insert(0, str(build))
    deep_ep = importlib.import_module("deep_ep")
    if Path(deep_ep.__file__).resolve() != (build / "deep_ep/__init__.py").resolve():
        raise RuntimeError("Unexpected DeepEP package origin.")
    native = Path(deep_ep._C.__file__).resolve()
    if not native.is_relative_to(root.resolve()):
        raise RuntimeError("Unexpected DeepEP native module origin.")
    if dist.is_initialized():
        raise RuntimeError("This isolated sample must own its one-rank process group.")
    x, ids = case.inputs
    layout = case.params["layout"]
    contiguous = layout.get("contiguous") if isinstance(layout, dict) else None
    if (not contiguous or contiguous.get("packing") != "tight"
            or contiguous.get("metadata_kind") != "physical_psum"
            or contiguous.get("alignment") != 1):
        raise UnsupportedCase("This sample is tight expert-major dispatch.")
    if x.dtype != torch.bfloat16:
        raise UnsupportedCase("The original unquantized dispatch requires BF16.")
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    experts = case.params["num_local_experts"]
    rendezvous = tempfile.TemporaryDirectory(prefix="classic-deepep-")
    buffer = None
    try:
        dist.init_process_group(
            backend="hccl", init_method=(Path(rendezvous.name) / "store").as_uri(),
            rank=0, world_size=1, timeout=timedelta(seconds=60),
        )
        buffer = deep_ep.EPBuffer(
            dist.group.WORLD, num_max_tokens_per_rank=tokens, hidden=hidden,
            num_topk=topk, deterministic=False, explicitly_destroy=True,
            num_cpu_timeout_secs=60, num_gpu_timeout_secs=60,
        )
    except Exception:
        if buffer is not None:
            buffer.destroy()
        deep_ep.destroy_all_managed_hccl_comm()
        if dist.is_initialized():
            dist.destroy_process_group()
        rendezvous.cleanup()
        raise

    def run(x, ids):
        # Rebuild routing each time: ids are inputs, not immutable setup data.
        rows, _, _, handle, _ = buffer.dispatch(
            x.contiguous(), topk_idx=ids.to(torch.int64).contiguous(),
            num_experts=experts, expert_alignment=1, do_expand=True, do_cpu_sync=True,
        )
        metadata = handle.recv_src_metadata
        if rows.shape != (tokens * topk, hidden) or metadata.shape != (tokens, topk + 2):
            raise RuntimeError("Original dispatch did not return every route for this sample.")
        # The native metadata already contains each route's output slot.
        # Reorder its rows by original token ID; no route sorting/recomputation.
        inverse = torch.empty((tokens, topk), device=x.device, dtype=torch.int32)
        inverse[metadata[:, -2].long()] = metadata[:, :topk].to(torch.int32)
        ends = handle.psum_num_recv_tokens_per_expert.to(torch.int32)
        return rows, ends, inverse.flatten()

    def cleanup():
        torch.npu.synchronize()
        buffer.destroy()
        deep_ep.destroy_all_managed_hccl_comm()
        dist.destroy_process_group()
        rendezvous.cleanup()

    files = [
        root / "deep_ep/buffers/ep.py", root / "deep_ep/comm/handle.py",
        root / "csrc/buffers/ep.hpp", root / "csrc/kernels/ep/dispatch.hpp",
        *list((root / "deep_ep/include/deep_ep/impls/ep").glob("*.hpp")),
    ]
    return Candidate(
        Implementation(run), "DeepEP-Ascend::EPBuffer.dispatch(world_size=1)",
        "Single-rank original dispatch, alignment=1; int32 IDs -> int64; native route metadata -> inverse slots.",
        source_hashes(Path(__file__), native, *files),
        limitations=("Original dispatch performs a host count wait; report complete wall time separately.",),
        timing_mode="host_roundtrip", cleanup=cleanup,
    )
