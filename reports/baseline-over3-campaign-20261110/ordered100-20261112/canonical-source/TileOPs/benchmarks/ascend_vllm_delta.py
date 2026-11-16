"""Original vLLM full GDN path with zero gate and standalone serving metadata."""
import ast
import hashlib
import importlib
from pathlib import Path
import sys
import types

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes, vllm_native,
)
from benchmarks.ascend_vllm_more import _original_module, _triton_bindings


def vllm_legacy_deltanet(case):
    import torch
    from benchmarks.api import Implementation

    _ops, binary, vendor = vllm_native()
    native_root = LIBRARIES / "vllm-ascend"
    source_root = native_root / "vllm_ascend/ops/triton/fla"
    bindings = _triton_bindings()
    envs = _original_module("vllm_ascend/envs.py", {})
    hardware = _original_module("vllm_ascend/ops/triton/triton_utils.py",
                                bindings | {"envs": envs})
    hardware.init_device_properties_triton()
    injected = types.ModuleType("_classic_vllm_fla_bindings")
    injected.__dict__.update(bindings)
    injected.__dict__.update({name: getattr(hardware, name) for name in
                              ("get_aicore_num", "get_vectorcore_num", "extract_slice", "insert_slice")})
    # Explicit standalone invocation: dense prefill and one process, no PCP.
    injected.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
    injected.get_pcp_group = lambda: types.SimpleNamespace(world_size=1, rank_in_group=0)
    injected.SUPPRESS_LEVEL = 0
    # Extract the original A5 dispatch methods, including their unchanged
    # launch options, without importing the serving framework.
    device_source = native_root / "vllm_ascend/device/device_op.py"
    device_tree = ast.parse(device_source.read_text(), str(device_source))
    original_class, = [n for n in device_tree.body
                       if isinstance(n, ast.ClassDef) and n.name == "A5DeviceAdaptor"]
    methods = [n for n in original_class.body if isinstance(n, ast.FunctionDef)
               and n.name in ("chunk_scaled_dot_kkt_fwd", "solve_tril_16x16")]
    assert len(methods) == 2
    dispatch_tree = ast.Module(body=[ast.ClassDef(
        name="DeviceOperator", bases=[], keywords=[], body=methods,
        decorator_list=[], type_params=[])], type_ignores=[])
    dispatch_globals = {"__name__": injected.__name__}
    exec(compile(ast.fix_missing_locations(dispatch_tree), str(device_source), "exec"),
         dispatch_globals)
    injected.DeviceOperator = dispatch_globals["DeviceOperator"]
    sys.modules[injected.__name__] = injected
    framework = {
        "vllm.triton_utils", "vllm_ascend.ops.triton.triton_utils",
        "vllm.distributed", "vllm.forward_context",
        "vllm.third_party.flash_linear_attention.ops.utils",
        "vllm_ascend.device.device_op",
    }
    originals = sorted(source_root.glob("*.py"))
    key = hashlib.sha256("".join(p.read_text() for p in originals).encode()).hexdigest()[:20]
    package_name = "_classic_vllm_fla_" + key
    cache = WORKSPACE / ".cache/ascend-classic" / package_name
    cache.mkdir(parents=True, exist_ok=True)
    generated = []
    for source in originals:
        if source.name == "__init__.py":
            continue
        tree = ast.parse(source.read_text(), str(source))
        # Only dependency binding names change. Kernel and host function bodies,
        # decorators and original operation sequence remain intact.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in framework:
                for entry in node.names:
                    if entry.name not in injected.__dict__:
                        raise RuntimeError(f"Unresolved original FLA import {entry.name}")
                node.module = injected.__name__
        target = cache / source.name
        target.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
        generated.append(target)
    package = types.ModuleType(package_name)
    package.__path__ = [str(cache)]
    sys.modules[package_name] = package
    fn = importlib.import_module(package_name + ".chunk").chunk_gated_delta_rule
    for module, symbol in (("chunk_scaled_dot_kkt", "chunk_scaled_dot_kkt_fwd_kernel"),
                           ("solve_tril", "solve_tril_16x16_kernel")):
        dispatch_globals[symbol] = getattr(
            importlib.import_module(package_name + "." + module), symbol)

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        # The original dense WY branch has an undefined i_t. Its existing
        # variable-length branch handles the same independent sequences.
        batch, length = q.shape[:2]
        qp, kp, vp = [t.reshape(1, batch * length, *t.shape[2:]).contiguous()
                      for t in (q, k, v)]
        bp = beta.reshape(1, batch * length, *beta.shape[2:]).contiguous()
        supplied = offsets if offsets is not None else offsets_cpu
        boundaries = (torch.arange(batch + 1, device=q.device, dtype=torch.int64) * length
                      if supplied is None else supplied.to(device=q.device, dtype=torch.int64))
        sequences = boundaries.numel() - 1
        gate = torch.zeros(bp.shape, device=q.device, dtype=torch.float32)
        # The native C++ wrapper derives final-state dtype from initial_state.
        # An explicit FP32 zero state is equivalent to the absent manifest state.
        state = initial if initial is not None else torch.zeros(
            (sequences, v.shape[2], q.shape[3], v.shape[3]),
            device=q.device, dtype=torch.float32)
        out, final = fn(qp, kp, vp, gate, bp, scale=case.params.get("scale"),
                        initial_state=state, output_final_state=True, head_first=False,
                        cu_seqlens=boundaries,
                        use_qk_l2norm_in_kernel=case.params.get("use_qk_l2norm_in_kernel", False))
        return out.reshape(v.shape), final

    kernels = [p for p in (native_root / "csrc").rglob("*")
               if p.is_file() and p.suffix in (".cpp", ".h") and
               ("chunk_delta" in str(p) or "chunk_gated" in str(p) or "chunk_fwd_o" in str(p))]
    deployed = [p for p in vendor.rglob("*") if p.is_file() and p.suffix in (".so", ".o", ".json")]
    return Candidate(
        Implementation(run), "vllm-ascend::chunk_gated_delta_rule(g=0)",
        "Original full GDN sequence; zero log gate, dense or manifest-ragged sequence metadata, "
        "original varlen path, explicit FP32 zero initial state, single-process PCP, final state enabled.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_vllm_more.py"),
                      binary, *originals, *generated, *kernels, *deployed,
                      native_root / "vllm_ascend/ops/triton/triton_utils.py",
                      native_root / "vllm_ascend/envs.py", device_source),
        limitations=("Original Triton and vLLM-owned native stages must all execute; no stock substitute.",),
    )


def vllm_deltanet(case):
    from benchmarks.ascend_vllm_kda import vllm_kda_deltanet
    return vllm_kda_deltanet(case)
