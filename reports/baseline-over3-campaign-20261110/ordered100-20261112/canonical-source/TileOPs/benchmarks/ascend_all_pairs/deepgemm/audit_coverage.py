"""Regenerate complete 205-op source coverage without tensor imports.

The shared inventory started at 1982 cases and was expanded to 2062 by the
supervisor while this worker ran. Always audit the complete current snapshot.
"""

import ast
from collections import Counter
import datetime
import hashlib
import json
from math import prod
from pathlib import Path
import re
import shutil

if __package__:
    from .layouts import convolution_plan
    from .reductions import reduction_plan
    from .pooling import pool_plan
    from .runtime import VENDOR, WORKSPACE, fingerprint, sha256
else:
    from layouts import convolution_plan
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from benchmarks.ascend_all_pairs.deepgemm.reductions import reduction_plan
    from benchmarks.ascend_all_pairs.deepgemm.pooling import pool_plan
    from runtime import VENDOR, WORKSPACE, fingerprint, sha256

HERE = Path(__file__).resolve().parent
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/deepgemm"
INVENTORY = EVIDENCE.parent / "inventory.json"
BASE = "ascend_baselines/DeepGEMM-Ascend/"


def reference(path, pattern=None):
    path = WORKSPACE / path
    lines = path.read_text().splitlines()
    line = 1 if pattern is None else next(i for i, value in enumerate(lines, 1) if pattern in value)
    return {"file": str(path.relative_to(WORKSPACE)), "line": line, "sha256": sha256(path)}


ROUTES = {
    "AvgPool1dFwdOp": ("average_pool", "bf16_gemm_nt", "TileOPs/workloads/pool.py", "class AvgPoolCall"),
    "AvgPool2dFwdOp": ("average_pool", "bf16_gemm_nt", "TileOPs/workloads/pool.py", "class AvgPoolCall"),
    "AvgPool3dFwdOp": ("average_pool", "bf16_gemm_nt", "TileOPs/workloads/pool.py", "class AvgPoolCall"),
    "AdaptiveAvgPool2dFwdOp": ("average_pool", "bf16_gemm_nt", "TileOPs/workloads/pool.py", "class AdaptiveAvgPool2dCall"),
    "MeanPoolingFwdOp": ("mean_pooling", "bf16_gemm_nt", "TileOPs/workloads/pool.py", "class MeanPoolingCallWorkload"),
    "SumFwdOp": ("reduction", "bf16_gemm_nt", "TileOPs/workloads/reduction.py", "class ReductionCall"),
    "MeanFwdOp": ("reduction", "bf16_gemm_nt", "TileOPs/workloads/reduction.py", "class ReductionCall"),
    "GemmFwdOp": ("gemm", "bf16_gemm_nt", "TileOPs/workloads/gemm.py", "class GemmWorkload"),
    "BmmFwdOp": ("bmm", "bf16_bmm", "TileOPs/workloads/gemm.py", "class BmmWorkload"),
    "GroupedGemmFwdOp": ("grouped_gemm", "bf16_gemm_nt", "TileOPs/workloads/gemm.py", "class GroupedGemmWorkload"),
    "GemmFP8FwdOp": ("fp8_gemm", "bf16_gemm_nt", "TileOPs/workloads/gemm.py", "class GemmFP8Workload"),
    "BmmFP8FwdOp": ("fp8_bmm", "bf16_bmm", "TileOPs/workloads/gemm.py", "class BmmFP8Workload"),
    "Conv1dFwdOp": ("convolution", "bf16_gemm_nt", "TileOPs/workloads/convolution.py", "class Conv1dWorkload"),
    "Conv2dFwdOp": ("convolution", "bf16_gemm_nt", "TileOPs/workloads/convolution.py", "class Conv2dWorkload"),
    "Conv3dFwdOp": ("convolution", "bf16_gemm_nt", "TileOPs/workloads/convolution.py", "class Conv3dWorkload"),
    "MoEGroupedGemmFwdOp": ("moe_grouped_gemm", "bf16_gemm_nt", "TileOPs/workloads/moe.py", "def ref_moe_grouped_gemm"),
    "SSDChunkCouplingFwdOp": ("ssd_chunk_coupling", "bf16_bmm", "TileOPs/workloads/mamba.py", "def ssd_chunk_coupling_fwd_ref"),
}

FAMILY_REASON = {
    "convolution": "BF16 convolution has a concrete patch-layout GEMM route; other operand dtypes require lossy narrowing and have no owned native implementation.",
    "elementwise": "The owned API has GEMM epilogues (identity, scalar alpha, bias, FP8 quantization, BF16 stochastic rounding), but no standalone elementwise contract. Applying an epilogue to a fabricated GEMM would introduce arithmetic beyond the allowed layout/API adaptations.",
    "reduction": "No standalone reduction/scan/statistics/softmax API. HC prenorm exposes GEMM plus squared sum only, which does not implement this complete reduction contract.",
    "norm": "No owned complete normalization forward/backward API, affine output or running-statistic outputs. HC prenorm returns a contraction and squared sum, not a normalized tensor.",
    "rope": "No rotation, frequency generation or RoPE API in the owned export surface.",
    "pool": "No pooling or pooling-index API. GEMM lowering would require introducing pooling arithmetic and cannot recover max-selection indices by a layout change.",
    "gemm": "Native kernels accept BF16/BF16, E4M3/E4M3 or E4M3/E2M1 FP4 with UE8M0 scales. No equivalent INT8 asymmetric or packed INT4 affine-dequantization path.",
    "attention": "Owned attention kernels emit weighted ReLU MQA index logits; no softmax, V contraction, backward attention or complete KV-cache operation. Those are not full-attention outputs.",
    "linear_attention": "No owned recurrent/chunk linear-attention kernel, decay/gate recurrence or output-state API; GEMM alone cannot produce the complete contract.",
    "quantization": "No standalone manifest quantizer/dequantizer. Scale layout conversion only repacks already exponent-only scales, and the GEMM FP8 epilogue produces E4M3 plus packed UE8M0 scales with a different output contract.",
    "moe": "MegaMoE owns fused FP8xFP4 dispatch/GEMMs/SwiGLU/combine with HCCL buffers and UE8M0 scales; it does not expose this BF16/FP16 routing/permutation/MLP contract. Quantizing the provided operands would be lossy; no standalone BF16 gated epilogue exists.",
    "mamba": "No owned SSM recurrence, exponential decay, scan or state propagation API. The chunk contraction route does not supply the extra mathematical stages or final states required here.",
    "sequence_modeling": "HC prenorm implements only D=float(A)@B.T and squared sum. It omits the gates, Sinkhorn iterations, residual/layer projections and other outputs of the full sequence-modeling contract.",
    "fft": "No Fourier transform or complex arithmetic API. A Fourier-matrix GEMM would change the allowed arithmetic and precision contract.",
    "transform": "No Hadamard transform API; no exact signed-transform path provided by the original native interface.",
    "sampling": "No probability mask, sorting/top-k selection or random sampling API. MQA logits computes scores only, and MegaMoE consumes preselected expert IDs.",
}


def semantic(op, family, p=None):
    if op in ("AvgPool1dFwdOp", "AvgPool2dFwdOp", "AvgPool3dFwdOp", "AdaptiveAvgPool2dFwdOp") and p is not None:
        try:
            pool_plan(op, p)
        except ValueError as exc:
            return "unsupported", "native_pool_precision_or_geometry", str(exc)
        return "supported_unverified", "native_gemm_pooling_api", "Pack complete original BF16 pooling windows and pass each original divisor to unchanged GEMM/Alpha; no split-K or extra input arithmetic. Numerical evidence remains separately version-bound."
    if op == "MeanPoolingFwdOp" and p is not None:
        shape, dtype = p["x"]
        if (dtype != "bfloat16" or p["output"][1] != dtype or p["accum_dtype"] != "float32"
                or "offsets" in p or "indices" in p or len(shape) != 4 or shape[1] % p["chunk_size"]):
            return "unsupported", "uniform_bf16_chunk_contract", "Only complete uniform BF16 chunks with original FP32 accumulation map to the admitted original GEMM/Alpha path."
        return "supported_unverified", "native_gemm_chunk_mean_api", "Permute complete original chunks into K; unchanged GEMM with native Alpha=1/original_chunk_size, full original BF16 output."
    if op in ("SumFwdOp", "MeanFwdOp") and p is not None:
        try:
            reduction_plan(p)
        except ValueError as exc:
            return "unsupported", "native_reduction_contract", str(exc)
        return "supported_unverified", "native_gemm_reduction_api", "Pack all original reduction axes; original BF16 GEMM with exact one-vector and native Alpha=1/K for Mean, BF16/FP32 output; untouched original numerical oracle still required."
    if op in ("CausalConv1dFwdOp", "CausalConv1dUpdateFwdOp"):
        return "unsupported", "missing_causal_silu_contract", "Current causal convolution cases require SiLU plus complete state output. Owned BF16 GEMM/bias can express the convolution contraction after layout packing, but exposes neither a standalone SiLU nor the complete fused stateful contract."
    if op == "GemmINT8W8A8FwdOp":
        return "unsupported", "missing_row_column_scale_epilogue", "INT8 operands themselves widen exactly to BF16, but the manifest requires independent arbitrary FP32 per-row and per-column scales. Native Alpha is scalar and native quantized scale input is UE8M0; no owned row/column epilogue implements this complete contract."
    if op == "FP8LightningIndexerFwdOp":
        return "unsupported", "mqa_weight_and_output_precision", "Native fp8_fp4_mqa_logits requires BF16 weights and returns BF16 logits; every manifest case supplies arbitrary FP32 weights and requires FP32 logits. Narrowing weights/output is not a lossless layout change. Native dimensions 64/128 themselves are not treated as a blocker."
    if op == "MoEGroupedGemmFP8FwdOp":
        return "unsupported", "arbitrary_fp32_block_scales", "Workload supplies arbitrary FP32 block scales in [0.5,1.5); native FP8 GEMM consumes packed UE8M0 exponent-only scales. Repeating 128-wide blocks into 32-wide blocks fixes granularity but cannot preserve the mantissas. Masked slabs alone would be adaptable."
    if op in ROUTES and p is not None:
        if op in ("GemmFP8FwdOp", "BmmFP8FwdOp"):
            if any(prod(p[name][0]) != 1 for name in ("scale_a", "scale_b")):
                return "unsupported", "arbitrary_fp32_block_scales", "Only per-tensor scales map to the original scalar Alpha epilogue. Arbitrary per-block FP32 scales cannot be represented losslessly by native UE8M0 scale input; block expansion fixes shape but not scale values."
            return "supported_unverified", "native_alpha_api", "E4M3 values widen exactly to BF16; current scalar scales map to native Alpha; native FP32 accumulation and final manifest output cast. Reassociated scaling and optional C=D bias accumulation still require the untouched numerical oracle."
        dtype_key = "input" if op.startswith("Conv") else "C_mat" if op == "SSDChunkCouplingFwdOp" else "a"
        if p[dtype_key][1] != "bfloat16":
            return "unsupported", "lossy_operand_narrowing", f"Input dtype {p[dtype_key][1]} cannot be represented losslessly by the owned BF16 GEMM input. ACLNN reference GEMMs are dependency calls, not a DeepGEMM-owned replacement."
        if op == "MoEGroupedGemmFwdOp" and p.get("activation") is not None:
            return "unsupported", "missing_gated_epilogue", "BF16 GEMM epilogues lack silu_and_mul; MegaMoE's fused activation requires quantized FP8/FP4 tensors and cannot preserve these BF16 inputs."
        if op.startswith("Conv"):
            convolution_plan(p)
        return "supported_unverified", "native_bf16_layout_route", "Concrete adapter invokes the owned BF16 GEMM/BMM through legal packing, transpose, padding/cropping and original API selection. Device execution and complete numerical/input-preservation checks are forbidden in this task."
    return "unsupported", "no_equivalent_owned_api", FAMILY_REASON[family]


def native_evidence(op, family):
    refs = [reference(BASE + "csrc/python_api.cpp", "gemm_api::register_apis")]
    if op in ROUTES:
        name, symbol, workload, marker = ROUTES[op]
        api = "einsum.hpp" if symbol == "bf16_bmm" else "gemm.hpp"
        refs += [reference(BASE + "csrc/apis/" + api, 'm.def("' + symbol + '"'),
                 reference(BASE + "csrc/jit_kernels/bf16_gemm.hpp", "static void launch_bf16_gemm"),
                 reference(BASE + "deep_gemm/include/deep_gemm/bf16_gemm.hpp", "void bf16_gemm_impl"),
                 reference(workload, marker)]
    if "FP8" in op or family == "quantization":
        refs += [reference(BASE + "csrc/apis/layout.hpp", "gran_k == 32"),
                 reference(BASE + "csrc/apis/gemm.hpp", "23-bit mantissa must be 0")]
    if family == "attention":
        refs += [reference(BASE + "csrc/apis/attention.hpp", "static torch::Tensor fp8_fp4_mqa_logits"),
                 reference(BASE + "csrc/apis/attention.hpp", '"Ascend MQA logits expects bf16 weights"')]
    if family == "moe":
        refs += [reference(BASE + "csrc/apis/mega_moe.hpp", 'mma_type == "fp8xfp4"'),
                 reference("TileOPs/workloads/moe.py", "def ref_moe_grouped_gemm")]
    if family in ("norm", "sequence_modeling", "reduction"):
        refs += [reference(BASE + "deep_gemm/tilelang_ops/hc_prenorm.py", "def tf32_hc_prenorm_gemm(a:")]
    if family in ("elementwise", "quantization", "moe") or op in ("GemmFP8FwdOp", "BmmFP8FwdOp"):
        refs += [reference(BASE + "csrc/apis/epilogue_class.hpp", "static void register_apis")]
    if op == "FP8LightningIndexerFwdOp":
        refs += [reference("TileOPs/workloads/attention/fp8_lightning_indexer.py", "class FP8LightningIndexerCall")]
    if op == "MoEGroupedGemmFP8FwdOp":
        refs += [reference("TileOPs/workloads/moe.py", "class MoEGroupedGemmFP8Workload")]
    return refs


def registry_entries():
    """Read literal registry keys/comprehension iterables, never import workloads."""
    result = {}
    directory = WORKSPACE / "TileOPs/benchmarks/_cases"
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "ENTRIES" for x in node.targets):
                for value in ast.walk(node.value):
                    if isinstance(value, ast.Constant) and isinstance(value.value, str) and re.fullmatch(r"\w+Op", value.value):
                        if value.value in result:
                            raise ValueError(f"Duplicate registry entry: {value.value}")
                        result[value.value] = {"file": str(path.relative_to(WORKSPACE)), "line": value.lineno, "sha256": sha256(path)}
    return result


def main():
    inventory = json.loads(INVENTORY.read_text())
    assert len(inventory["operators"]) == 205 and len(inventory["cases"]) >= 1982
    entries = registry_entries()
    deployment = json.loads((HERE / "deployment_probe.json").read_text())
    deployed = deployment["status"] == "host_built_unverified"
    by_op = {op["op"]: [] for op in inventory["operators"]}
    for case in inventory["cases"]:
        by_op[case["op"]].append(case)
    bindings = []
    for path in sorted((VENDOR / "csrc/apis").glob("*.hpp")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r'm\.(?:def|attr)\("([^"]+)"', line)
            if match:
                bindings.append({"symbol": match.group(1), "file": str(path.relative_to(WORKSPACE)), "line": number, "sha256": sha256(path)})
    source_audit = {"schema": "deepgemm-source-audit/v1", "source_sha256": fingerprint(),
                    "native_bindings": bindings,
                    "excluded_dependency_apis": ["aclnn_bf16_gemm_*", "aclnn_fp8_fp4_gemm_*", "cublaslt_gemm_* (aliases to ACLNN)"],
                    "excluded_third_party_math": ["third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py (fused quantized output, not standalone manifest activation)"]}
    (HERE / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")
    operators, cases = [], []
    for operator in inventory["operators"]:
        op, family = operator["op"], operator["family"]
        manifest = "TileOPs/" + operator["manifest"]
        manifest_ref = reference(manifest, op + ":")
        assert manifest_ref["sha256"] == inventory["manifest_sha256"][manifest], manifest
        refs = [manifest_ref, *native_evidence(op, family)]
        adapter = None
        if op in ROUTES:
            module = "pooling.py" if ROUTES[op][0] in ("average_pool", "mean_pooling") else "reductions.py" if op in ("SumFwdOp", "MeanFwdOp") else "adapters.py"
            adapter = reference(str((HERE / module).relative_to(WORKSPACE)), "def " + ROUTES[op][0] + "(") | {"factory": ROUTES[op][0]}
        local_cases = []
        for case in by_op[op]:
            state, code, reason = semantic(op, family, case["params"])
            effective = "supported_unverified" if state == "supported_unverified" and deployed else "deployment_blocked" if state == "supported_unverified" else "unsupported"
            local_cases.append({
                "op": op, "case_id": case["case_id"], "status": effective,
                "workload_factory_available": op in entries,
                "semantic_status": state, "reason_code": code, "reason": reason,
                "deployment_status": "host_built_unverified" if deployed else "deployment_blocked" if state == "supported_unverified" else "not_applicable",
                "deployment_receipt": str((HERE / "deployment_probe.json").relative_to(WORKSPACE)) if state == "supported_unverified" else None,
                "adapter": adapter if state == "supported_unverified" else None,
                "source_evidence": refs,
                "numerical_status": "not_run_cpu_only", "numerical_receipt": None,
                "input_protection_receipt": None, "native_launch_attribution_receipt": None,
                "params_sha256": hashlib.sha256(json.dumps(case["params"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            })
        states = Counter(c["status"] for c in local_cases)
        supported = sum(c["semantic_status"] == "supported_unverified" for c in local_cases)
        missing = op not in entries
        assert len(local_cases) == operator["inventoried_case_count"]
        fallback_state, fallback_code, fallback_reason = semantic(op, family)
        operators.append({"op": op, "family": family, "case_count": len(local_cases),
                          "status": "missing_workload_factory" if missing else "supported_unverified" if supported and deployed else "deployment_blocked" if supported else "unsupported",
                          "semantic_status": "supported_unverified" if supported else fallback_state,
                          "case_status_counts": dict(states), "supported_case_count": supported,
                          "registered_provider": op in ROUTES, "adapter": adapter,
                          "workload_factory": entries.get(op), "missing_workload_factory": missing,
                          "reason": "Case-level constraints and deployment blockers are recorded for every case." if supported else fallback_reason,
                          "source_evidence": refs, "numerical_receipt": None})
        cases.extend(local_cases)
    missing = [op["op"] for op in operators if op["missing_workload_factory"]]
    summary = {"operators": len(operators), "manifest_cases": len(cases), "registered_providers": len(ROUTES),
               "operator_status_counts": dict(Counter(op["status"] for op in operators)),
               "case_status_counts": dict(Counter(c["status"] for c in cases)),
               "case_semantic_status_counts": dict(Counter(c["semantic_status"] for c in cases)),
               "missing_workload_factories": missing, "device_runs": 0, "verified": 0, "numerical_failures": 0}
    coverage = {"schema": "deepgemm-all-pairs-coverage/v1", "library": "DeepGEMM-Ascend",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "inventory": {"file": str(INVENTORY.relative_to(WORKSPACE)), "sha256": sha256(INVENTORY)},
                "scope_change": {"assigned_case_count": 1982, "current_case_count": len(cases),
                                 "cause": "Shared inventory expanded by supervisor during worker execution; all current rows audited."},
                "status_definitions": {
                    "supported_unverified": "Concrete owned-library adapter; semantic/source review only; no device/numerical claim.",
                    "deployment_blocked": "Adapter has a supported_unverified semantic path but current host/compiler deployment is unavailable; not mathematical unsupported.",
                    "unsupported": "No equivalent native mathematical/dtype contract under permitted adaptations; reason and source evidence supplied.",
                    "missing_workload_factory": "Operator inventoried but shared case registry has no factory; inventoried rows are not executable tests.",
                    "numerical_failure": "Reserved for actual device oracle failures; none observed because all execution is forbidden.",
                    "verified": "Requires complete manifest outputs, input preservation and owned native launch receipt; none in this CPU-only task.",
                },
                "historical_results": "No classic verified result was promoted into this ARM64/CANN9.1 audit.",
                "summary": summary, "operators": operators, "cases": cases}
    (HERE / "coverage.json").write_text(json.dumps(coverage, indent=2, allow_nan=False) + "\n")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    snapshot = EVIDENCE / ("inventory_" + sha256(INVENTORY)[:16] + ".json")
    if not snapshot.exists():
        shutil.copy2(INVENTORY, snapshot)
    progress_path = HERE / "progress.json"
    if progress_path.exists() and not (EVIDENCE / "progress_initial.json").exists():
        shutil.copy2(progress_path, EVIDENCE / "progress_initial.json")
    progress = {"schema": "deepgemm-all-pairs-progress/v1", "worker": "546a72b6d7b44cbe",
                "owner": "allpairs-remote/deepgemm", "phase": "implemented_awaiting_static_audit",
                "updated_at": coverage["created_at"], "cpu_only": True, "summary": summary,
                "deployment_blockers": deployment["blockers"],
                "remaining": ["Complete and save static audit", "Future authorized compatible host build and complete original manifest device checks"],
                "oracle_issues": [], "no_memory_fork_merge": True,
                "shared_harness_issues": {"missing_workload_factories": missing,
                    "timing": "Grouped, MoE grouped and scalar FP8 adapters require host_roundtrip timing; all metadata reads occur inside run."}}
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    for name in ("coverage.json", "progress.json", "source_audit.json", "deployment_probe.json"):
        shutil.copy2(HERE / name, EVIDENCE / name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
