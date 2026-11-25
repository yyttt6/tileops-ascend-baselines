"""Measure the eager upstream workload on Ascend using production CI profile windows."""
import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

from benchmarks.ascend.guard import require_reservation

require_reservation()

import torch
import torch_npu
from benchmarks.api import Implementation, cases
from benchmarks.baselines import private_inputs
from measured_correctness import measure_profile_and_verify
from workloads.device import set_run_device

def run(op, case, output):
    seed = int.from_bytes(hashlib.sha256(f"{op}::{case.id}".encode()).digest()[:4], "little")
    torch.manual_seed(seed)
    row = {"op": op, "case_id": case.id, "params": case.params, "seed": seed, "provider": "torch-eager"}
    positions = {"NGramHashFwdOp": (1,), "GatedPoolCompressFwdOp": (3, 4)}.get(op, ())
    if op in {"FP4QuantPerGroupFwdOp", "RoPECosSinFwdOp"} and case.arguments["inplace"]:
        positions = (0,)
    try:
        impl = private_inputs(case.torch, case.inputs, *positions) if positions else Implementation(run=case.torch)
        args = case.inputs if impl.args is None else impl.args
        correctness, measurement = measure_profile_and_verify(
            case, impl, torch, torch_npu, args, output / "profiles" / op / case.id, mode="isolated"
        )
        row.update(status="passed", correctness=correctness, measurement=measurement,
                   warmup=3, repeats=15, flush_bytes=1 << 30)
    except Exception:
        row.update(status="failed", error=traceback.format_exc())
    return row

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("op")
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    torch.npu.set_device(0)
    set_run_device("npu:0")
    options.output.mkdir(exist_ok=True, parents=True)
    failed = 0
    with (options.output / (options.op + ".jsonl")).open("x") as rows:
        for case in cases(options.op):
            row = run(options.op, case, options.output)
            rows.write(json.dumps(row, default=str) + "\n")
            rows.flush()
            print(case.id, row["status"], flush=True)
            failed += row["status"] == "failed"
            case._release()
            gc.collect()
    torch.npu.synchronize()
    raise SystemExit(bool(failed))
