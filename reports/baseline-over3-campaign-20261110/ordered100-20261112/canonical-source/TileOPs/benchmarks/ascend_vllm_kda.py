"""Use vLLM's complete original KDA entries with zero decay for DeltaNet."""
import ast
import hashlib
import importlib
from pathlib import Path
import sys
import types

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
from benchmarks.ascend_vllm_more import _original_module, _triton_bindings


def vllm_kda_deltanet(case):
    import torch
    from benchmarks.api import Implementation

    q, k, v = case.inputs[:3]
    if q.shape != k.shape or q.shape[2:] != v.shape[2:]:
        raise UnsupportedCase("Original KDA output allocation requires matched Q/K/V heads and feature widths.")
    bindings = _triton_bindings()
    envs = _original_module("vllm_ascend/envs.py", {})
    hardware = _original_module("vllm_ascend/ops/triton/triton_utils.py", bindings | {"envs": envs})
    hardware.init_device_properties_triton()
    injected = types.ModuleType("_classic_vllm_kda_bindings")
    injected.__dict__.update(bindings)
    injected.cdiv = bindings["triton"].cdiv
    injected.next_power_of_2 = bindings["triton"].next_power_of_2
    injected.get_vectorcore_num = hardware.get_vectorcore_num
    sys.modules[injected.__name__] = injected
    native = LIBRARIES / "vllm-ascend"
    original_root = native / "vllm_ascend/ops/triton/kda"
    originals = sorted(original_root.glob("*.py"))
    digest = hashlib.sha256("".join(f.read_text() for f in originals).encode()).hexdigest()[:20]
    name = "_classic_vllm_kda_" + digest
    cache = WORKSPACE / ".cache/ascend-classic" / name
    cache.mkdir(parents=True, exist_ok=True)
    generated = []
    imports = {"vllm.triton_utils", "vllm.utils.math_utils", "vllm_ascend.ops.triton.triton_utils"}
    for original in originals:
        if original.name == "__init__.py":
            continue
        tree = ast.parse(original.read_text(), str(original))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in imports:
                if any(n.name not in injected.__dict__ for n in node.names):
                    raise RuntimeError(f"Unresolved original KDA import in {original}")
                node.module = injected.__name__
        target = cache / original.name
        target.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
        generated.append(target)
    package = types.ModuleType(name)
    package.__path__ = [str(cache)]
    sys.modules[name] = package
    module = importlib.import_module(name + ".kda")
    recurrent = case.inputs[0].shape[1] == 1
    fn = module.fused_recurrent_kda if recurrent else module.chunk_kda

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        batch, length = q.shape[:2]
        original_shape = v.shape
        supplied = offsets if offsets is not None else offsets_cpu
        pad_heads = (supplied is not None and q.shape[2] == 6
                     and case.params.get("use_qk_l2norm_in_kernel", False))
        if pad_heads:
            # Heads are independent; the original L2 reduction is over K only.
            # Keep sequence offsets, feature dimensions and native stages intact.
            import torch.nn.functional as F
            q, k, v = [F.pad(t, (0, 0, 0, 2)) for t in (q, k, v)]
            beta = F.pad(beta, (0, 2))
            if initial is not None:
                initial = F.pad(initial, (0, 0, 0, 0, 0, 2))
        boundaries = (torch.arange(batch + 1, device=q.device, dtype=torch.int64) * length
                      if supplied is None else supplied.to(device=q.device, dtype=torch.int64))
        sequences = boundaries.numel() - 1
        qp, kp = [t.reshape(1, batch * length, *t.shape[2:]).contiguous() for t in (q, k)]
        # Original chunk_kda uses v as its output buffer; retain manifest inputs.
        vp = v.reshape(1, batch * length, *v.shape[2:]).clone().contiguous()
        bp = beta.reshape(1, batch * length, beta.shape[-1]).contiguous()
        state = (torch.zeros((sequences, v.shape[2], v.shape[3], q.shape[3]),
                             device=q.device, dtype=torch.float32)
                 if initial is None else initial.transpose(-1, -2).clone().contiguous())
        gate = torch.zeros(qp.shape, device=q.device, dtype=torch.float32)
        kwargs = dict(scale=case.params.get("scale"), initial_state=state,
                      use_qk_l2norm_in_kernel=case.params.get("use_qk_l2norm_in_kernel", False),
                      cu_seqlens=boundaries)
        if recurrent:
            # Packed one-token sequences give the original per-token final-state
            # allocation exactly one state for each independent manifest sequence.
            kwargs.update(inplace_final_state=False, compute_gate=False, sigmoid_beta=False)
        else:
            kwargs["output_final_state"] = True
        output, final = fn(qp, kp, vp, gate, bp, **kwargs)
        if pad_heads:
            output = output[..., :original_shape[2], :].contiguous()
            final = final[:, :original_shape[2], :, :].contiguous()
        return output.reshape(original_shape), final.transpose(-1, -2).contiguous()

    return Candidate(
        Implementation(run), "vllm-ascend.kda." + ("fused_recurrent_kda" if recurrent else "chunk_kda") + "(g=0)",
        "Original complete KDA entry with zero channel-wise log decay equals ungated delta recurrence; preserve beta, native L2 flag and FP32 states; transpose state layout, pack sequence boundaries and clone the original in-place value buffer.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_vllm_delta.py"),
                      Path(__file__).with_name("ascend_vllm_more.py"), *originals, *generated,
                      native / "vllm_ascend/ops/triton/triton_utils.py", native / "vllm_ascend/envs.py"),
        limitations=("All original complete kernel stages and default launch/auto-tuning settings unchanged; no use of the quarantined GDN native output kernel.",),
    )
