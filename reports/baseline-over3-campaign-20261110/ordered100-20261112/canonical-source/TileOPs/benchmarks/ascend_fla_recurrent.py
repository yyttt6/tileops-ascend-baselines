"""Original FLA recurrent KDA for one-token BF16 DeltaNet cases."""
import importlib
import os
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
from benchmarks.ascend_extra_baselines import _owned_native


def fla_recurrent_deltanet(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    q, k, v, beta, initial, offsets, offsets_cpu = case.inputs
    if q.shape[1] != 1 or q.dtype != torch.bfloat16 or offsets is not None or offsets_cpu is not None:
        raise UnsupportedCase("This original recurrent entry pilot uses independent dense BF16 one-token sequences.")
    dim, value_dim = q.shape[-1], v.shape[-1]
    if max(dim, value_dim) > 128:
        raise UnsupportedCase("Only feature zero-padding to the original128 is used.")
    root = LIBRARIES / "flash-linear-attention-npu"
    original_package = root / "torch_custom/fla_npu/fla_npu"
    task = WORKSPACE / ".cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu-recurrent-kda"
    installed = Path(os.environ.get("ASCEND_CLASSIC_FLA_PACKAGE", str(task / "python_binary/fla_npu"))).resolve()
    if installed != (task / "python_binary/fla_npu").resolve():
        raise RuntimeError("Recurrent KDA requires its separately audited original package.")
    files = sorted(original_package.rglob("*.py"))
    for original in files:
        if (installed / original.relative_to(original_package)).read_bytes() != original.read_bytes():
            raise RuntimeError("FLA recurrent deployment/source mismatch.")
    binary = installed / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    _owned_native(binary, ["aclnnRecurrentKda", "aclnnRecurrentKdaGetWorkspaceSize"])
    os.environ["FLA_NPU_STABLE_ABI"] = "ctypes"
    public = importlib.import_module("fla_npu.ops.ascendc")
    if Path(public.__file__).resolve() != (installed / "ops/ascendc/__init__.py").resolve():
        raise RuntimeError("FLA recurrent resolved outside the audited package.")
    fn = public.npu_recurrent_kda
    scale = case.params.get("scale")
    scale = dim ** -0.5 if scale is None else scale

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        qp, kp, vp = F.pad(q, (0, 128-dim)), F.pad(k, (0, 128-dim)), F.pad(v, (0, 128-value_dim))
        state = None if initial is None else F.pad(initial, (0, 128-value_dim, 0, 128-dim)).clone()
        gate = torch.zeros((*q.shape[:2], v.shape[2], 128), device=q.device, dtype=torch.float32)
        out, final = fn(qp, kp, vp, gate, beta, state,
                        layout="BSND", scale=scale, output_final_state=True,
                        inplace_final_state=False, state_v_first=False,
                        use_qk_l2norm_in_kernel=case.params.get("use_qk_l2norm_in_kernel", False),
                        use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        if public.BACKENDS.get("npu_recurrent_kda") != "ctypes":
            raise RuntimeError("FLA recurrent did not use the original owned ctypes backend.")
        return out[..., :value_dim].contiguous(), final[..., :dim, :value_dim].contiguous()

    native = root / "fla/ops/ascendc/kda/recurrent_kda"
    sources = sorted(f for f in native.rglob("*") if f.suffix in (".cpp", ".h", ".hpp"))
    return Candidate(
        Implementation(run), "fla_npu.ops.ascendc.npu_recurrent_kda(g=0)",
        "Original complete recurrent KDA with zero channel decay equals DeltaNet; original beta/L2/scale and FP32 state preserved, only feature padding/cropping and private state copy.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_extra_baselines.py"), binary, *files, *sources),
        limitations=("Separate original native deployment and launch attribution required; BF16 decode only, full outputs checked at original tolerances.",),
    )
