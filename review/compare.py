"""CPU-only full-raw comparison; a numeric target is separate from qualification."""
import argparse
import importlib.util
import math
from pathlib import Path
import statistics

from review.common import HERE, TILEOPS_COMMIT, read, sha, write_new


def audit(path, side, replay=True):
    row = read(path)
    if row.get("schema") != "baseline-review-measurement-v1" or row.get("side") != side:
        raise ValueError("Expected a review.bench measurement with the requested side")
    if row["status"] != "measured_pending_identity_review" or not row["inputs_preserved"]:
        raise ValueError("Measurement incomplete")
    for key in ["full_original_check", "measured_output_check"]:
        check = row[key]
        if not check["total_outputs"] or check["checked_outputs"] != check["total_outputs"] or check["unchecked_reason"]:
            raise ValueError("Original full-output correctness evidence incomplete")
    timing = row["timing"]
    samples, profiles = timing["latency_samples_ms"], timing["retained_profiles"]
    if row["protocol"] != "ascend950-isolated-ffts-v3" or len(samples) != 15 or len(profiles) != 15:
        raise ValueError("Require the same isolated protocol and all15 raw windows")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples):
        raise ValueError("Invalid sample")
    if not math.isclose(statistics.mean(samples) * 1000, row["device_us"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Mean differs from saved samples")
    if replay:
        spec = importlib.util.spec_from_file_location("review_full32", HERE / "support/tools/kernel_task_profile_full32.py")
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        for sample, profile in zip(samples, profiles, strict=True):
            trace = Path(profile["profile_path"])
            if parser.trace_digest(trace) != profile["profile_trace_sha256"]:
                raise ValueError("Raw profile changed or missing; keep its original path locally")
            piece = parser.parse_profile(trace, 1)
            if not piece["profile_named_task_coverage_complete"] or abs(piece["latency_ms"] - sample) > 1e-9:
                raise ValueError("Full raw replay differs")
    if side == "tileops":
        identity_path = path.parent / "TILEOPS_GENERATED_IDENTITY.json"
        if sha(identity_path) != row["tileops_generated_identity_sha256"]:
            raise ValueError("Public executing identity changed")
        identity = read(identity_path)
        if not identity["generated_identity_complete"]:
            raise ValueError("Public executing source/binary identity incomplete")
        for name, digest in identity["generated_sources_sha256"].items():
            if sha(path.parent / "tileops-generated" / name) != digest:
                raise ValueError("Retained public source changed")
        for record in identity["generated_binaries"].values():
            if sha(path.parent / "tileops-generated" / record["retained"]) != record["sha256"]:
                raise ValueError("Retained public binary changed")
    return row


def compare(baselines, denominator, review=None):
    d = audit(denominator, "tileops")
    if d["tileops_commit"] != TILEOPS_COMMIT:
        raise ValueError("Wrong TileOPs revision")
    if len({str(p.resolve()) for p in baselines}) != len(baselines):
        raise ValueError("Repeated path is not an independent process")
    rows = [audit(p, "baseline") for p in baselines]
    processes = [tuple(sorted(row["runtime"]["process_identity"].items())) for row in rows]
    if len(processes) != len(set(processes)):
        raise ValueError("Two result files from the same process do not establish independent repetition")
    for row in rows:
        for key in ["key", "payload_sha256", "contract_sha256", "protocol"]:
            if row[key] != d[key]:
                raise ValueError(f"Noncomparable {key}")
        for key in ["hostname", "cube", "vector", "device_name", "precision"]:
            if row["runtime"][key] != d["runtime"][key]:
                raise ValueError(f"Different device/profile field: {key}")
        if row["runtime"]["process_identity"]["boot_id"] != d["runtime"]["process_identity"]["boot_id"]:
            raise ValueError("Different boot; establish a new matched denominator")
    means = [r["device_us"] for r in rows]
    ratio = max(means) / d["device_us"]
    numeric = ratio <= 3.0
    repeat = 2.9 <= ratio <= 3.0 and len(rows) < 2
    reviewed = False
    if review:
        evidence = read(review)
        checks = ["same_physical_device_driver_firmware", "approved_runtime_profiles",
                  "executing_native_and_jit_binary_identity", "full_api_and_unchanged_precision",
                  "no_interference_or_profiler_loss"]
        if not all(evidence.get(k) is True for k in checks):
            raise ValueError("Identity review is incomplete")
        expected = {str(p.resolve()): sha(p) for p in [*baselines, denominator]}
        if evidence.get("result_sha256") != expected:
            raise ValueError("Identity review does not bind these exact result files")
        if not evidence.get("reviewer") or not evidence.get("physical_device_identity"):
            raise ValueError("Review needs reviewer and actual physical device identity")
        reviewed = True
    return {"key": d["key"], "baseline_process_means_us": means,
            "baseline_worst_mean_us": max(means), "tileops_mean_us": d["device_us"],
            "ratio_baseline_over_tileops": ratio, "numeric_target_met": numeric,
            "independent_repeat_required": repeat, "identity_review_complete": reviewed,
            "accepted": numeric and not repeat and reviewed,
            "status": "accepted_local_comparison" if numeric and not repeat and reviewed else
                      "repeat_required" if repeat else "identity_review_required" if numeric else "continue_optimization",
            "scope_note": "One exact local case, not historical revalidation, all665 completion, publication or CI adoption"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", nargs="+", type=Path, required=True)
    p.add_argument("--tileops", type=Path, required=True)
    p.add_argument("--identity-review", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.tileops, args.identity_review)
    write_new(args.output, result)
    print(__import__("json").dumps(result, indent=2))


if __name__ == "__main__":
    main()
