"""Guarded, isolated validation of real cross-case compiled-artifact consumption."""

import argparse
import ctypes
import dataclasses
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def measure_and_release(item, measure):
    """Drop per-case device inputs and callable even if measurement fails."""
    try:
        return measure(item)
    finally:
        item.pop("args", None)
        item.pop("compiled", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--claim", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["serial", "pipeline"], required=True)
    p.add_argument("--indices", default="0,4,6")
    p.add_argument("--windows", type=int, choices=[1, 15], default=15)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with open("/run/lock/ascend-npu/device-0.lock") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("device reservation required")
    # Set private caches before importing either compiler; no production cache writes.
    for key, directory in [
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", "ascendc"),
        ("ASCEND_CACHE_PATH", "ascend"),
    ]:
        os.environ[key] = str(a.output / "cache" / directory)
    sys.path[:0] = [
        str(a.root / "frozen/source/src"),
        str(a.root / "frozen/source"),
        str(a.root / "frozen"),
    ]
    report = {
        "kind": "experimental_cross_case_pipeline",
        "mode": a.mode,
        "cases": [],
        "start_mono_ns": time.monotonic_ns(),
    }
    backend = None

    def event(name, **values):
        with (a.output / "timeline.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "event": name,
                        "wall_ns": time.time_ns(),
                        "mono_ns": time.monotonic_ns(),
                        **values,
                    }
                )
                + "\n"
            )

    try:
        import torch
        import torch_npu
        from benchmarks.api import cases
        from workloads.device import set_run_device
        from workloads.numerics import compare_outputs
        from tools import kernel_task_profile_full32 as parser

        sys.modules["tools.kernel_task_profile"] = parser
        from tools import measure_baselines as core
        from staged_backend import StagedBackend

        torch.set_num_threads(2)
        torch._dynamo.config.suppress_errors = False

        def cv(x, device):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, tuple):
                return tuple(cv(v, device) for v in x)
            if isinstance(x, list):
                return [cv(v, device) for v in x]
            if isinstance(x, dict):
                return {k: cv(v, device) for k, v in x.items()}
            return x

        def zero(x):
            if isinstance(x, torch.Tensor):
                return torch.zeros_like(x)
            if isinstance(x, tuple):
                return tuple(zero(v) for v in x)
            if isinstance(x, list):
                return [zero(v) for v in x]
            if isinstance(x, dict):
                return {k: zero(v) for k, v in x.items()}
            return x

        def checked(output, expected, policy):
            cpu = cv(output, "cpu")
            result = compare_outputs(cpu, expected, policy)
            assert (
                result.checked_outputs == result.total_outputs
                and result.total_outputs > 0
            )
            try:
                compare_outputs(zero(cpu), expected, policy)
            except AssertionError:
                pass
            else:
                raise RuntimeError("complete zero output accepted")
            return dataclasses.asdict(result)

        rows = json.loads(a.claim.read_text())["rows"]
        items = []
        set_run_device("cpu")
        for i in map(int, a.indices.split(",")):
            torch.manual_seed(0)
            row = rows[i]
            case = next(c for c in cases(row["op"]) if c.id == row["case_id"])
            assert json.loads(json.dumps(case.params)) == row["case"]["params"]
            inputs = case.inputs
            expected = case.reference(
                *tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
            )
            items.append(
                {"row": row, "case": case, "cpu_inputs": inputs, "expected": expected}
            )
        torch.npu.set_device(0)
        torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        properties = torch.npu.get_device_properties(0)
        assert (properties.cube_core_num, properties.vector_core_num) == (32, 64)
        set_run_device("npu")
        binary = a.root / "flush-rebuild/build01"
        assert (
            hashlib.sha256((binary / "flush.aibin").read_bytes()).hexdigest()
            == "693be2fb78b6c749ec2afee81e1ed3d8c0de58ea3397daaeaeef06588cc2e865"
        )
        dll = ctypes.CDLL(str(binary / "flush.so"))
        dll.init.argtypes = [ctypes.c_char_p]
        dll.init.restype = ctypes.c_int
        dll.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.call.restype = ctypes.c_int
        assert dll.init(str(binary / "flush.aibin").encode()) == 0

        def flush(buffer):
            assert (
                dll.call(buffer.data_ptr(), torch.npu.current_stream().npu_stream) == 0
            )

        buffer = torch.empty(268435456, dtype=torch.int32, device="npu")
        buffer.fill_(1)
        flush(buffer)
        torch.npu.synchronize()
        assert torch.count_nonzero(buffer.cpu()).item() == 0
        core._flush_kernel_cache[0] = (flush, buffer)
        backend = StagedBackend(a.output / "builds", workers=2)
        original_window = core._time_one_isolated_repeat

        def observed(*args, **kwargs):
            backend.check()
            result = original_window(*args, **kwargs)
            backend.check()
            return result

        core._time_one_isolated_repeat = observed

        def prepare(i):
            item = items[i]
            item["args"] = cv(item["cpu_inputs"], "npu")
            event("prepare_begin", case=i)
            item["stage"], item["compiled"] = backend.prepare(
                "case" + str(i), item["case"].reference, item["args"]
            )
            event("prepare_end", case=i)
            assert (
                item["stage"]["deferred_before_kernel_launch"]
                and item["stage"]["kernels"]
            ), "fixture did not exercise deferred AscendC compilation"

        def finish(i):
            item = items[i]
            event("consume_begin", case=i)
            output = backend.finish(item["stage"], item["compiled"], item["args"])
            checked(output, item["expected"], item["case"].verification)
            for actual, expected_input in zip(item["args"], item["cpu_inputs"]):
                if isinstance(actual, torch.Tensor):
                    assert torch.equal(actual.cpu(), expected_input), (
                        "input mutated during preparation/consumption"
                    )
            event("consume_end", case=i)

        def measure(i, item):
            directory = a.output / ("case" + str(i))
            directory.mkdir()
            row = item["row"]
            case = item["case"]
            record = {
                "op": row["op"],
                "case": row["case_id"],
                "contract_sha256": row["contract_sha256"],
                "windows": a.windows,
                "timings": {},
                "negative_controls": {},
                "timed_checks": {},
                "kind": report["kind"],
                "sources": {},
            }
            for label, fn in [
                ("reference", case.reference),
                ("checker", compare_outputs),
                ("parser", parser.parse_profile),
                ("core", core._time_reference),
            ]:
                file = Path(inspect.getsourcefile(fn))
                record["sources"][label] = {
                    "path": str(file),
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            for variant, fn in [
                ("eager", case.reference),
                ("compile", item["compiled"]),
            ]:
                os.environ["TILEOPS_PROFILE_DIR"] = str(
                    directory / variant / "profiles"
                )
                last = {}

                def capture(*args):
                    value = fn(*args)
                    last["out"] = value
                    return value

                before = {
                    str(k): dict(v) for k, v in torch._dynamo.utils.counters.items()
                }
                event("measurement_begin", case=i, variant=variant)
                timing = core._time_reference(
                    torch, torch_npu, capture, item["args"], warmup=3, repeats=a.windows
                )
                event("measurement_end", case=i, variant=variant)
                after = {
                    str(k): dict(v) for k, v in torch._dynamo.utils.counters.items()
                }
                record.setdefault("graph_counters", {})[variant] = {
                    "before": before,
                    "after": after,
                    "unchanged": before == after,
                }
                (directory / "admission.json").write_text(json.dumps(record, indent=2))
                assert before == after, (
                    "unexpected graph compilation during measurement"
                )
                record["timed_checks"][variant] = checked(
                    last["out"], item["expected"], case.verification
                )
                record["negative_controls"][variant] = {
                    "complete_output_zeroed": True,
                    "rejected": True,
                }
                assert len(timing["retained_profiles"]) == a.windows
                for profile, sample in zip(
                    timing["retained_profiles"], timing["latency_samples_ms"]
                ):
                    path = Path(profile["profile_path"])
                    assert parser.trace_digest(path) == profile["profile_trace_sha256"]
                    parsed = parser.parse_profile(path, 1)
                    assert parsed["profile_named_task_coverage_complete"]
                    assert abs(parsed["profile_kernel_us"] / 1000 - sample) < 1e-9
                record["timings"][variant] = {
                    "measurement": timing,
                    "offline_replay": f"{a.windows}/{a.windows} passed",
                }
                (directory / "admission.json").write_text(json.dumps(record, indent=2))
            report["cases"].append(
                {"op": row["op"], "case": row["case_id"], "path": str(directory)}
            )

        prepare(0)
        finish(0)
        for i, item in enumerate(items):
            if i + 1 < len(items):
                prepare(i + 1)
                if a.mode == "serial":
                    finish(i + 1)
            measure_and_release(item, lambda current: measure(i, current))
            if i + 1 < len(items) and a.mode == "pipeline":
                finish(i + 1)
        report["status"] = "passed_pending_independent_guard_and_raw_audit"
    except Exception as error:
        report.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        try:
            if backend:
                backend.close()
                backend.check()
        except Exception as error:
            report.update(status="failed", cleanup_error=repr(error))
            raise
        finally:
            if backend:
                report["backend"] = backend.report()
            report["end_mono_ns"] = time.monotonic_ns()
            report["elapsed_seconds"] = (
                report["end_mono_ns"] - report["start_mono_ns"]
            ) / 1e9
            (a.output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
