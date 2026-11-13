"""Run frozen main-source operators sequentially, with one guard reservation per operator."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ascend950_snapshot import seal
from nightly_snapshot import build_cache, validate

HERE = Path(__file__).resolve().parent


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def identity(commit):
    files = [
        HERE / "device_batch.py",
        HERE / "nightly_support/session_profile.py",
        HERE / "precompile.py",
        HERE / "nightly_device.py",
        HERE / "measured_correctness.py",
        HERE / "golden_cache.py",
        HERE / "golden_cache_policy.json",
        *sorted((HERE / "nightly_support").rglob("*.py")),
    ]
    return dict(
        commit=commit,
        measurement_files={
            str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        toolchain=hashlib.sha256(
            Path("/opt/tileops-ci/server2-20261030/config/toolchain-v1.lock.json").read_bytes()
        ).hexdigest(),
    )


def select_operators(selection, requested):
    stale = selection.get("unmapped_registrations", [])
    if stale:
        raise ValueError(f"Ascend registrations absent from the current manifest: {stale}")
    ops = selection["operators"] if requested == "all" else requested.split(",")
    missing = sorted(set(ops) - set(selection["operators"]))
    if missing:
        raise ValueError(f"Requested operators have no discovered Ascend implementation: {missing}")
    if not ops or len(ops) != len(set(ops)):
        raise ValueError("Operator selection must be nonempty and contain no duplicates")
    return ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--operators", default="all")
    p.add_argument("--resume", type=Path)
    p.add_argument("--previous-snapshot", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    a.source = a.source.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert os.environ["GITHUB_REPOSITORY"] == "yyttt6/TileOPs.github.io"
    assert os.environ["GITHUB_REF"] == "refs/heads/main"
    commit = subprocess.check_output(
        ["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "-C", str(a.source), "status", "--porcelain"], text=True
    ).strip()
    from device_batch import groups
    from device_batch import run as run_batch

    groups([], a.batch_size)
    current_identity = dict(
        identity(commit),
        batch_size=a.batch_size,
        timer=os.environ.get("ASCEND950_TIMER_MODE", "auto"),
    )
    write(out / "identity.json", current_identity)
    prior = {}
    if a.resume:
        a.resume = a.resume.resolve()
        assert json.loads((a.resume / "identity.json").read_text()) == current_identity, (
            "Checkpoint environment mismatch"
        )
        prior = json.loads((a.resume / "receipts.json").read_text())
    start = time.monotonic()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    timings = []

    def run(op, target, attempt=0):
        command = [
            "npu-guard",
            "run",
            "--owner",
            f"nightly/{os.environ['GITHUB_RUN_ID']}/{op}/{attempt}",
            "--wait",
            "1800",
            "--timeout",
            "7200",
            "--",
            sys.executable,
            str(HERE / "nightly_device.py"),
            "--source",
            str(a.source),
            "--output",
            str(target),
            "--op",
            op,
            "--verify-measured-output",
        ]
        t = time.monotonic()
        with (out / f"{op}-attempt{attempt}.log").open("x") as log:
            r = subprocess.run(command, cwd=a.source, stdout=log, stderr=subprocess.STDOUT)
        timings.append(
            dict(
                op=op,
                attempt=attempt,
                guard_exit_code=r.returncode,
                elapsed_seconds=time.monotonic() - t,
            )
        )
        write(
            out / "timing.json",
            dict(started_at=started, elapsed_seconds=time.monotonic() - start, stages=timings),
        )
        return r.returncode

    assert run("__list__", out / "selection.json") == 0
    selection = json.loads((out / "selection.json").read_text())
    ops = select_operators(selection, a.operators)
    write(out / "plan.json", dict(commit=commit, operators=ops, started_at=started))
    sys.path.insert(0, str(HERE / "nightly_support"))

    receipts = {}
    measured = []
    failures = []
    frozen_sources = None
    live = {}
    batch_number = 0
    for op_index, op in enumerate(ops):
        saved = prior.get(op, {})
        if saved.get("guard_exit_code") == 0 and saved.get("result", {}).get("status") == "passed":
            assert saved["hardware"] == selection["hardware"], "Checkpoint hardware mismatch"
            assert not saved["result"]["failed"] and not saved["result"]["source_changed"]
            directory = saved.get("attempts", [{}])[saved.get("selected_attempt", 0)].get(
                "directory", op
            )
            origin = (a.resume / directory).resolve()
            assert origin.is_relative_to(a.resume)
            target = out / op
            shutil.copytree(origin, target)
            code, result = 0, saved["result"]
            receipts[op] = dict(
                saved,
                resumed_from=str(a.resume),
                attempts=[dict(attempt=0, directory=op)],
                selected_attempt=0,
            )
            print("RESUME", op, "CPU replay of retained raw profiles", flush=True)
        else:
            if op not in live:
                available = [
                    name
                    for name in ops[op_index:]
                    if name not in live
                    and not (
                        prior.get(name, {}).get("guard_exit_code") == 0
                        and prior.get(name, {}).get("result", {}).get("status") == "passed"
                    )
                ]
                batch = groups(available, a.batch_size)[0]
                label = f"batch-{batch_number:04d}"
                batch_number += 1
                report = run_batch(
                    a.source,
                    [dict(op=name, output=str(out / name)) for name in batch],
                    out,
                    label,
                    f"nightly/{os.environ['GITHUB_RUN_ID']}/{label}",
                    timer=current_identity["timer"],
                )
                timings.append(
                    dict(
                        batch=batch,
                        guard_exit_code=report["code"],
                        elapsed_seconds=report["seconds"],
                    )
                )
                write(
                    out / "timing.json",
                    dict(
                        started_at=started, elapsed_seconds=time.monotonic() - start, stages=timings
                    ),
                )
                completed = {item["op"] for item in report["worker"].get("records", [])}
                for name in batch:
                    target = out / name
                    if name not in completed and report["code"] == 0:
                        continue  # An earlier operator failed; start untouched work in a fresh process.
                    result = (
                        json.loads((target / "result.json").read_text())
                        if (target / "result.json").exists()
                        else {}
                    )
                    receipt = dict(
                        guard_exit_code=report["code"],
                        result=result,
                        hardware=json.loads((target / "hardware.json").read_text())
                        if (target / "hardware.json").exists()
                        else {},
                        measurement_run_id=os.environ["GITHUB_RUN_ID"],
                        batch=label,
                        guard_metadata=report["guard"],
                        guard_receipt=report["guard_receipt"],
                        guard_sha256=report["guard_sha256"],
                        attempts=[dict(attempt=0, directory=name)],
                        selected_attempt=0,
                    )
                    live[name] = receipt
                    receipts[name] = receipt
                write(
                    out / "receipts.json", receipts
                )  # Only completed guards produce reusable checkpoints.
            target = out / op
            receipt = live[op]
            code, result = receipt["guard_exit_code"], receipt["result"]
            receipts[op] = receipt

        if code or result.get("status") != "passed":
            failures.append(op)
        else:
            sources = json.loads((target / "sources.json").read_text())
            if frozen_sources is None:
                frozen_sources = sources
            assert sources == frozen_sources
            case_rows = [
                json.loads(line) for line in (target / "results.jsonl").read_text().splitlines()
            ]
            assert {r["case_id"] for r in case_rows} == set(selection["cases"][op])
            assert len(case_rows) == len(selection["cases"][op])
            for line in (target / "results.jsonl").read_text().splitlines():
                row = json.loads(line)
                from complete_profile import replay_coverage

                assert result.get("selection_scope", "all-cases") == "all-cases"
                coverage = replay_coverage(row["measurement"])
                row["raw_replayed"] = True
                row["measurement"] = {
                    k: v
                    for k, v in row["measurement"].items()
                    if k
                    not in (
                        "profile_coverage_adapter",
                        "profile_coverage_evidence",
                        "retained_profiles",
                    )
                }
                row["coverage"] = [
                    {"complete_sha256": c["complete_sha256"], "result": c["result"]}
                    for c in coverage
                ]
                row["workload"] = dict(
                    op=op,
                    case_id=row["case_id"],
                    family=row["family"],
                    params=row["params"],
                    tensors={
                        k: v
                        for k, v in row["params"].items()
                        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list)
                    },
                )
                measured.append(row)
        write(out / "receipts.json", receipts)
        print(op, result.get("passed"), result.get("failed"), "guard", code, flush=True)
    assert not failures, f"Failed operators; no publication: {failures}"
    snapshot = out / "snapshot"
    snapshot.mkdir()
    base = json.loads((HERE.parent / "data/baseline-cache/snapshot.json").read_text())
    inherited = None
    if a.previous_snapshot and (a.previous_snapshot / "meta.json").exists():
        from ascend950_snapshot import validate as validate_prior
        from baseline_release import merge_latest_release

        inherited, _ = validate_prior(a.previous_snapshot)
        previous = json.loads((a.previous_snapshot / "baseline-cache.json").read_text())
        release = json.loads((HERE.parent / "data/baseline-cache/latest-release.json").read_text())
        base = merge_latest_release(previous, base, release)
    # Bind unpaired workloads to their family from the frozen source manifest.
    for row in measured:
        row["workload"]["family"] = row["family"]
    date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta = dict(
        schema_version=2,
        publisher="server2-physical32",
        publisher_repository="https://github.com/yyttt6/TileOPs.github.io",
        status="passed",
        measurement_protocol="ascend950-adaptive-ffts-v4",
        commit=commit,
        date=date,
        operators=ops,
        case_ids={op: selection["cases"][op] for op in ops},
        fresh_cases=len(measured),
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        elapsed_seconds=time.monotonic() - start,
        source_branch="ascend950-v0.1.0",
        contract_identity_version="v010-reviewed-aliases-v4",
        selection_scope=a.operators,
        registered_operators=len(selection["operators"]),
        inherited_snapshot=None
        if inherited is None
        else {k: inherited[k] for k in ("run_id", "commit", "date", "fresh_cases")},
    )
    for name, value in [
        ("meta", meta),
        ("sources", frozen_sources),
        ("measurements", measured),
        ("receipts", receipts),
        ("baseline-before", base),
        (
            "baseline-cache",
            build_cache(
                base,
                measured,
                commit,
                date,
                use_v010_aliases=True,
                fresh_protocol=meta["measurement_protocol"],
            ),
        ),
    ]:
        write(snapshot / f"{name}.json", value)
    seal(snapshot)
    validate(snapshot)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
