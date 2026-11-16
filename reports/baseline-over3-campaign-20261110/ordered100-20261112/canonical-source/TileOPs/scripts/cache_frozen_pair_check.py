#!/usr/bin/env python3
"""Verify one all-pairs library candidate against one exact manifest case.

Run through npu-guard and the selected baseline-run environment. One process per
case isolates import/compiler/device failures. --profile additionally uses the
existing workspace FFTS protocol; a correctness-only pass is never a timing win.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial, wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent)]


def require_guard():
    """Refuse accidental naked runs before importing any device runtime."""
    metadata = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
    child = metadata.get("child", {})
    if metadata.get("phase") != "running" or child.get("pid") != os.getpgrp():
        raise RuntimeError("Run this complete command inside host npu-guard.")


def add_edge_check(torch, case, impl):
    """Same compiled shape, new inputs, cancellation/extremes and signed zero.

    This also catches wrappers which cached input-dependent copies in build().
    Workload.verify remains the authoritative random-input contract check.
    """
    x, y = case.inputs
    values_x = [-0., 0., 1., -1., float("inf"), -float("inf"), float("nan"),
                torch.finfo(x.dtype).tiny, torch.finfo(x.dtype).max]
    values_y = [-0., -0., -1., 1., -float("inf"), float("inf"), 1.,
                -torch.finfo(y.dtype).tiny, torch.finfo(y.dtype).max]
    before = [t.clone() for t in (x, y)]
    try:
        x.view(-1)[:len(values_x)].copy_(torch.tensor(values_x, device=x.device, dtype=x.dtype))
        y.view(-1)[:len(values_y)].copy_(torch.tensor(values_y, device=y.device, dtype=y.dtype))
        # A custom OPP may also intercept Torch's NPU Add. CPU evaluation
        # makes the oracle independent of the candidate's device kernel.
        expected = case.reference(x.cpu(), y.cpu())
        actual = impl.run(x, y)
        torch.npu.synchronize()
        actual = actual.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        zero = expected == 0
        # The standard equality check treats +0 and -0 as equal.
        torch.testing.assert_close(torch.signbit(actual)[zero], torch.signbit(expected)[zero])
        torch.testing.assert_close(x.view(-1)[:len(values_x)],
                                   torch.tensor(values_x, device=x.device, dtype=x.dtype),
                                   rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(y.view(-1)[:len(values_y)],
                                   torch.tensor(values_y, device=y.device, dtype=y.dtype),
                                   rtol=0, atol=0, equal_nan=True)
    finally:
        x.copy_(before[0])
        y.copy_(before[1])
        torch.npu.synchronize()
    return {"changed_inputs": True, "extremes": True, "nonfinite": True,
            "signed_zero": True, "independent_cpu_reference": True}


def fp8_edge_check(torch, case, impl):
    """Exercise the scale floor on the same shape with an independent oracle."""
    from workloads.numerics import assert_quantized

    x, = case.inputs
    probe = torch.zeros_like(x)
    rows = probe.reshape(-1, x.shape[-1])
    pattern = torch.linspace(-1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    for row, amplitude in enumerate((0., 1e-6, 1e-4, 1.), start=0):
        rows[row].copy_(pattern * amplitude)
    original = probe.cpu()
    expected = case.reference(original)
    actual = impl.run(probe)
    torch.npu.synchronize()
    assert_quantized(tuple(value.cpu() for value in actual), expected)
    torch.testing.assert_close(probe.cpu(), original, rtol=0, atol=0)
    return {"changed_inputs": True, "zero_groups": True, "tiny_groups": True,
            "scale_floor": True, "independent_cpu_reference": True}


def rope_table_diagnostics(torch, case, impl, library):
    """Independently check rotation with identical coefficients; report table drift."""
    from workloads.numerics import Request, verify
    from workloads.rope import ref_rope_neox, ref_rope_non_neox, rope_frequency_tables

    x, = case.inputs
    assert x.ndim in (2, 4) and case.params["rope_layout"] in ("neox", "interleaved")
    rotation = ref_rope_neox if case.params["rope_layout"] == "neox" else ref_rope_non_neox
    kwargs = dict(head_dim=x.shape[-1], seq_len=x.shape[0] if x.ndim == 2 else x.shape[1],
                  base=case.params.get("base", 10000.), dtype=x.dtype)
    device_tables = rope_frequency_tables(**kwargs, device=x.device)
    host_tables = rope_frequency_tables(**kwargs, device="cpu")
    torch.npu.synchronize()
    copied_tables = tuple(t.cpu() for t in device_tables)
    original = x.cpu()
    actual = impl.run(x)
    torch.npu.synchronize()
    actual = actual.cpu()
    torch.testing.assert_close(x.cpu(), original, rtol=0, atol=0)
    result = {"input_preserved": True, "tables": {}}
    for name, device, host in zip(("cos", "sin"), copied_tables, host_tables, strict=True):
        result["tables"][name] = {
            "different_elements": int((device != host).sum()),
            "total_elements": host.numel(),
            "max_abs_difference": float((device.float() - host.float()).abs().max()),
        }
    references = {
        "cpu_rotation_with_device_tables": lambda v: rotation(v, *copied_tables),
        "cpu_rotation_with_cpu_tables": lambda v: rotation(v, *host_tables),
    }
    for name, reference in references.items():
        try:
            checks = verify(reference, (original,), evidence=case.verification,
                            requests={library: Request(lambda _: actual, (original,),
                                                       preserve_inputs=True)})
            result[name] = {"passed": True, **asdict(checks[library])}
        except AssertionError as exc:
            result[name] = {"passed": False, "error": str(exc)}
    # main() requires the same-coefficient CPU check as well as the manifest's
    # original target-device oracle. Cross-device table drift is reported.
    return result


def cpu_reference(torch, reference, inputs):
    """Keep custom OPP interception out of the oracle's arithmetic."""
    device = next(value.device for value in inputs if isinstance(value, torch.Tensor))

    def move(value, destination):
        if isinstance(value, torch.Tensor):
            return value.to(destination)
        if isinstance(value, tuple):
            return tuple(move(item, destination) for item in value)
        if isinstance(value, list):
            return [move(item, destination) for item in value]
        if isinstance(value, dict):
            return {key: move(item, destination) for key, item in value.items()}
        return value

    def run(*args):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(threads, 4))
            expected = reference(*move(args, "cpu"))
        finally:
            torch.set_num_threads(threads)
        return move(expected, device)

    return run


def verify_fp64_on_cpu(torch, case, observed_run, impl_args, reset, library, records):
    """Device dispatch only: original reference and full validator, no math changes."""
    from workloads.device import run_device, set_run_device
    from workloads.numerics import Request, verify
    def host(value):
        if isinstance(value, torch.Tensor):
            return value.to('cpu')
        if isinstance(value, tuple): return tuple(host(v) for v in value)
        if isinstance(value, list): return [host(v) for v in value]
        if isinstance(value, dict): return {k:host(v) for k,v in value.items()}
        return value
    def reference(*inputs):
        previous, threads = run_device(), torch.get_num_threads()
        try:
            set_run_device('cpu')
            torch.set_num_threads(min(threads, 4))
            return case.reference(*host(inputs))
        finally:
            set_run_device(previous)
            torch.set_num_threads(threads)
    def candidate(*inputs):
        return host(observed_run(*inputs))
    return verify(observe_call(torch, reference, records), case.inputs,
                  evidence=case.verification,
                  requests={library:Request(candidate,impl_args,reset,preserve_inputs=True)},
                  preserve_reference_inputs=True)

def tensor_metadata(torch, value):
    """Describe actual tensor layout without reading payloads or synchronizing."""
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "device": str(value.device), "layout": str(value.layout),
                "elements": value.numel(),
                "stride": list(value.stride()) if value.layout == torch.strided else None}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__,
                "items": [tensor_metadata(torch, item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(key): tensor_metadata(torch, item)
                                           for key, item in value.items()}}
    return {"kind": "none" if value is None else type(value).__name__}


def observe_call(torch, run, records):
    """Retain per-call metadata from the exact callable used by verification."""
    @wraps(run)
    def observed(*args, **kwargs):
        record = {"inputs": tensor_metadata(torch, args),
                  "keyword_inputs": tensor_metadata(torch, kwargs), "returned": False}
        records.append(record)
        result = run(*args, **kwargs)
        record.update(outputs=tensor_metadata(torch, result), returned=True)
        return result
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu-fp64-reference", action="store_true",
                        help="Run original GDN/KDA FP64 reference and full validator on CPU; candidate remains on NPU.")
    args = parser.parse_args()
    if args.cpu_fp64_reference and args.op not in ("GDNFwdOp", "KDAFwdOp"):
        parser.error("--cpu-fp64-reference is limited to GDNFwdOp and KDAFwdOp")
    require_guard()
    from benchmarks.ascend_all_pairs.registry import LIBRARY_MODULES

    if args.library not in LIBRARY_MODULES:
        parser.error(f"Unknown baseline library: {args.library}")
    package = ROOT / "benchmarks/ascend_all_pairs" / LIBRARY_MODULES[args.library]
    provider_sources = {
        str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Do not overwrite a receipt: {args.output}")
    row = dict(op=args.op, case_id=args.case_id, library=args.library,
               started_at=datetime.now(timezone.utc).isoformat(), profile_requested=args.profile,
               verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               providers_sha256=hashlib.sha256(
                   (ROOT / "benchmarks/ascend_all_pairs/registry.py").read_bytes()).hexdigest(),
               adapter_sha256=hashlib.sha256(
                   (ROOT / "benchmarks/ascend_baselines.py").read_bytes()).hexdigest(),
               sequence_metadata_sha256=hashlib.sha256(
                   (ROOT / "workloads/sequence_metadata.py").read_bytes()).hexdigest())
    started = time.monotonic()
    rc = 1
    stage = "runtime_import"
    torch = None
    candidate = None
    oracle_sources = {}
    try:
        import torch
        import torch_npu
        from benchmarks.api import cases
        from benchmarks.ascend_baselines import UnsupportedCase
        from benchmarks.ascend_all_pairs.registry import candidate_factory
        from tools.baseline_protocol import seed_case
        from workloads.device import set_run_device
        from workloads.numerics import Request, verify

        set_run_device("npu")
        row["input_seed"] = seed_case(torch, args.op, args.case_id)
        matching = [c for c in cases(args.op) if c.id == args.case_id]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one manifest case; got {len(matching)}")
        case = matching[0]
        row["case_snapshot"] = {
            "op": args.op, "case_id": case.id, "parameters": case.params,
            "declared_tensors": case._call.tensors,
            "declared_input_names": list(case._call.signature.inputs),
            "declared_output_names": list(case._call.signature.outputs),
        }
        row["observed_calls"] = {"reference": [], "candidate": []}
        # Keep the actual workload/binder sources beside each device receipt.
        # A change to another workload should not invalidate this case.
        for obj in (type(case.workload), case.reference, case._entry.workload,
                    case._entry.binder, case._entry.inputs):
            if obj is None:
                continue
            while isinstance(obj, partial):
                obj = obj.func
            filename = inspect.getsourcefile(obj)
            if filename:
                path = Path(filename).resolve()
                if path.is_relative_to(ROOT.parent):
                    oracle_sources[str(path.relative_to(ROOT.parent))] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
        for path in (ROOT / "workloads/numerics.py", ROOT / "benchmarks/api.py",
                     ROOT / "workloads/workload_base.py"):
            oracle_sources[str(path.relative_to(ROOT.parent))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
        if args.op in ("DeltaNetChunkFwdOp", "DeltaNetChunkBwdOp", "GLAChunkBwdOp"):
            path = ROOT / "workloads/linear_attention/chunk_cpu.py"
            oracle_sources[str(path.relative_to(ROOT.parent))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
        row["oracle_source_sha256"] = oracle_sources
        cpu_fft_inputs = None
        if args.op == "FFTC2CFwdOp":
            # This backend cannot generate complex random inputs or evaluate
            # all complex reference/checking operators on NPU. Keep the frozen
            # case's shape/dtype and run the candidate with device inputs.
            set_run_device("cpu")
            try:
                cpu_fft_inputs = tuple(case.workload.gen_inputs())
            finally:
                set_run_device("npu")
            case.__dict__["inputs"] = tuple(x.to("npu") for x in cpu_fft_inputs)
            row["input_generation_device"] = "cpu"
            row["reference_device"] = "cpu"
        stage = "build"
        row["tvm_module_before_build"] = getattr(sys.modules.get("tvm"), "__file__", None)
        try:
            candidate = candidate_factory(case, args.library)()
        except UnsupportedCase as exc:
            row.update(status="unsupported", reason=str(exc))
            rc = 0
        else:
            # Fingerprint this library's integration package in addition to the
            # vendor hashes declared by Candidate.
            row["provider_source_sha256"] = provider_sources
            row["runtime_host"] = {"hostname": __import__("socket").gethostname(),
                                   "architecture": __import__("platform").machine()}
            row.update(api=candidate.api, adaptation=candidate.adaptation,
                       source_sha256=candidate.source_sha256, count_copies=True,
                       limitations=candidate.limitations, timing_mode=candidate.timing_mode)
            row["source_sha256"]["TileOPs/workloads/numerics.py"] = hashlib.sha256(
                (ROOT / "workloads/numerics.py").read_bytes()).hexdigest()
            impl = candidate.implementation
            observed_run = observe_call(torch, impl.run, row["observed_calls"]["candidate"])
            impl_args = case.inputs if impl.args is None else impl.args
            stage = "verify"
            if args.op == "RoPEFwdOp":
                row["rope_table_diagnostics"] = rope_table_diagnostics(
                    torch, case, impl, args.library)
                if not row["rope_table_diagnostics"]["cpu_rotation_with_device_tables"]["passed"]:
                    raise AssertionError("Independent CPU rotation with identical coefficients failed.")
                row["independent_cpu_rotation_with_device_tables"] = True
            if cpu_fft_inputs is not None:
                def fft_device_call(*cpu_args):
                    device_args = tuple(x.to("npu") for x in cpu_args)
                    actual = observed_run(*device_args)
                    torch.npu.synchronize()
                    for device_arg, original in zip(device_args, cpu_args, strict=True):
                        torch.testing.assert_close(device_arg.cpu(), original, rtol=0, atol=0)
                    return actual.cpu()
                observed_reference = observe_call(torch, case.reference, row["observed_calls"]["reference"])
                checks = verify(observed_reference, cpu_fft_inputs, evidence=case.verification,
                                requests={args.library: Request(fft_device_call, cpu_fft_inputs,
                                                                preserve_inputs=True)})
            elif args.cpu_fp64_reference:
                row['reference_device'] = 'cpu'
                row['validator_device'] = 'cpu'
                row['independent_cpu_reference'] = True
                row['reference_engine'] = 'unchanged original FP64 reference and full validator on CPU'
                row['reference_device_reason'] = 'Explicit --cpu-fp64-reference; NPU does not support original DT_DOUBLE arithmetic'
                checks = verify_fp64_on_cpu(torch,case,observed_run,impl_args,impl.reset,args.library,row['observed_calls']['reference'])
            else:
                reference = case.reference
                if ((args.library in ("ops-math", "ops-nn", "ops-transformer")
                     and args.op not in ("DeltaNetFwdOp", "RoPEFwdOp"))
                        or args.op == "AdaptiveMaxPool2dFwdOp"):
                    reference = cpu_reference(torch, reference, case.inputs)
                    row["reference_device"] = "cpu"
                    row["independent_cpu_reference"] = True
                    if args.op == "AdaptiveMaxPool2dFwdOp":
                        # The NPU reference faults on the original BF16
                        # nondivisible case before invoking the candidate.
                        # CPU executes the same reference and output contract.
                        row["reference_engine"] = "original manifest adaptive_max_pool2d on CPU"
                        row["reference_device_reason"] = "NPU reference VEC SIMT stack overflow observed before candidate invocation"
                elif args.op == "RoPEFwdOp":
                    # The manifest computes its frequency tables on x.device.
                    # CPU table generation rounds differently before rotation;
                    # retain the original target-device contract and require
                    # the independent same-coefficient CPU check above.
                    row["reference_device"] = "npu"
                    row["reference_engine"] = "original manifest RoPE reference"
                    row["independent_cpu_reference"] = False
                elif args.op == "DeltaNetFwdOp":
                    # The manifest's FLA reference runs Triton kernels and has
                    # no CPU implementation. Preserve that original oracle;
                    # it does not dispatch the custom ChunkKdaFwd OPP kernel.
                    row["reference_device"] = "npu"
                    row["reference_engine"] = "original manifest FLA delta_rule"
                    row["independent_cpu_reference"] = False
                    if case.inputs[5] is not None or case.inputs[6] is not None:
                        from benchmarks.ascend_delta_reference import packed_delta_reference
                        reference = packed_delta_reference(case, row)
                        row["reference_engine"] = "original dense FLA per logical sequence; independently audited with CPU FP64 recurrence"
                        row["independent_cpu_reference"] = True
                        row["reference_adaptation"] = "packed_to_independent_dense_sequences"
                if any(isinstance(t, torch.Tensor) and t.dtype == torch.float8_e4m3fn
                       for t in case.inputs):
                    from workloads.numerics import _unchanged
                    # Exercise all FP8 bit patterns, then a real mutation, before
                    # using this platform compatibility path for input checking.
                    probe = torch.arange(256, dtype=torch.int32, device="npu").to(torch.uint8).view(torch.float8_e4m3fn)
                    snapshot = probe.clone()
                    assert _unchanged([probe], [snapshot])
                    probe.view(torch.uint8)[0] = 56  # +0 -> +1, a genuine value change.
                    assert not _unchanged([probe], [snapshot])
                    row["fp8_input_preservation_selfcheck"] = {
                        "all_256_bit_patterns_unchanged": True, "mutation_detected": True}
                reference = observe_call(torch, reference, row["observed_calls"]["reference"])
                checks = verify(reference, case.inputs, evidence=case.verification,
                                requests={args.library: Request(observed_run, impl_args, impl.reset,
                                                                preserve_inputs=True)},
                                preserve_reference_inputs=bool(row.get("reference_adaptation")))
            check = checks[args.library]
            row["verification"] = asdict(check)
            if not check.checked_outputs or check.checked_outputs != check.total_outputs:
                raise ValueError("Candidate did not verify every output.")
            if args.op == "AddFwdOp":
                stage = "edge_check"
                row["edge_check"] = add_edge_check(torch, case, impl)
            if args.op == "FP8QuantFwdOp":
                stage = "edge_check"
                row["edge_check"] = fp8_edge_check(torch, case, impl)
            row["inputs_preserved"] = True
            row["status"] = "verified"
            if args.profile:
                if candidate.timing_mode != "device_async":
                    raise ValueError("Host-roundtrip candidates require a separate wall-time report.")
                stage = "profile"
                from tools.measure_baselines import _time_reference
                row["timing"] = _time_reference(torch, torch_npu, impl.run, impl_args,
                                                 warmup=3, repeats=15, reset=impl.reset)
                row["status"] = "measured"
            rc = 0
    except Exception as exc:
        row.update(status="failed", stage=stage, error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc())
    finally:
        if torch is not None and hasattr(torch, "npu") and torch.npu.is_initialized():
            try:
                torch.npu.synchronize()
            except Exception as exc:
                row.update(status="device_sync_failed", sync_error=str(exc))
                rc = 78
        if candidate is not None and candidate.cleanup is not None:
            try:
                candidate.cleanup()
            except Exception as exc:
                row.update(status="cleanup_failed", cleanup_error=str(exc))
                # A cleanup exception must not hide a preceding device fault.
                if rc != 78:
                    rc = 1
        final_sources = {
            str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
        row["provider_source_changed"] = final_sources != provider_sources
        if row["provider_source_changed"]:
            row["provider_source_sha256_after"] = final_sources
            row["status"] = "source_changed"
            rc = 1
        oracle_after = {
            relative: hashlib.sha256((ROOT.parent / relative).read_bytes()).hexdigest()
            if (ROOT.parent / relative).is_file() else None
            for relative in oracle_sources
        }
        row["oracle_source_changed"] = oracle_after != oracle_sources
        if row["oracle_source_changed"]:
            row["oracle_source_sha256_after"] = oracle_after
            row["status"] = "source_changed"
            rc = 1
        row["wall_seconds"] = time.monotonic() - started
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n")
        print(json.dumps({key: row.get(key) for key in
                          ("op", "case_id", "library", "status", "stage", "error")}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
