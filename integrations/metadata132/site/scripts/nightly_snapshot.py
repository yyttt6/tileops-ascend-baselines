"""Validate platform-default nightly measurements and preserve historical cache rows."""

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

PROTOCOL = "ascend950-isolated-ffts-v3"
COVERAGE = "v3-distinct-ffts-plus-missing-named-aicpu-v1"
SESSION_PROTOCOL = "ascend950-marker-session-v4"
SESSION_COVERAGE = "v4-bracketed-ffts-plus-named-aicpu-v1"
ADAPTIVE_PROTOCOL = "ascend950-adaptive-ffts-v4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_params(op, params):
    """Normalize the cache's JSON-safe infinity encoding for vector norm ord only."""
    normalized = dict(params)
    if op == "VectorNormFwdOp":
        order = normalized.get("ord")
        if isinstance(order, float) and math.isinf(order):
            normalized["ord"] = "Infinity" if order > 0 else "-Infinity"
    return normalized


def build_cache(
    base,
    measured,
    commit,
    date,
    *,
    use_v010_aliases=False,
    fresh_protocol=PROTOCOL,
    alias_version="v010-reviewed-aliases-v4",
):
    if base.get("baseline_release"):
        return build_versioned_cache(
            base,
            measured,
            commit,
            date,
            use_v010_aliases=use_v010_aliases,
            fresh_protocol=fresh_protocol,
            alias_version=alias_version,
        )
    data = copy.deepcopy(base)
    fresh = {(r["op"], r["case_id"]): r for r in measured}
    assert len(fresh) == len(measured)
    consumed = set()
    for row in data["rows"]:
        key = (row["op"], row["case"])
        if key not in fresh:
            continue
        m = fresh[key]
        assert contract_params(row["op"], row["workload"]["params"]) == contract_params(
            m["op"], m["params"]
        ), f"Changed cached contract: {key}"
        latency = m["measurement"]["latency_ms"] * 1000
        row.update(
            tileops_latency_us=latency,
            ratio=row["latency_us"] / latency,
            tileops_verified=True,
            tileops_samples=15,
            tileops_protocol=m["measurement"].get("measurement_protocol", PROTOCOL),
            tileops_commit=commit,
            tileops_date=date,
            tileops_measurement_id=hashlib.sha256(
                json.dumps(m, sort_keys=True).encode()
            ).hexdigest(),
        )
        consumed.add(key)
    existing = {(r["op"], r["case"]): r for r in data.get("tileops_rows", [])}
    for key, m in fresh.items():
        if key not in consumed:
            existing[key] = dict(
                op=key[0],
                case=key[1],
                workload=m["workload"],
                latency_us=m["measurement"]["latency_ms"] * 1000,
                samples=15,
                verified=True,
                commit=commit,
                date=date,
                protocol=m["measurement"].get("measurement_protocol", PROTOCOL),
            )
    data["tileops_rows"] = [v for k, v in existing.items() if k not in consumed]
    data["frozen_at"] = date
    data["nightly"] = dict(
        commit=commit,
        date=date,
        fresh_cases=len(measured),
        native_baselines="Preserved original saved v2; not remeasured",
        fresh_protocol=fresh_protocol,
        clock_policy="platform-default",
    )
    return data


def build_versioned_cache(
    base,
    measured,
    commit,
    date,
    *,
    use_v010_aliases=False,
    fresh_protocol=PROTOCOL,
    alias_version="v010-reviewed-aliases-v4",
):
    """Update each measured contract separately; changed definitions never share ratios."""
    if alias_version == "v010-reviewed-aliases-v1":
        from baseline_contracts_v1 import comparison_identity
    elif alias_version == "v010-reviewed-aliases-v2":
        from baseline_contracts_v2 import comparison_identity
    elif alias_version == "v010-reviewed-aliases-v3":
        from baseline_contracts_v3 import comparison_identity
    else:
        assert alias_version == "v010-reviewed-aliases-v4"
        from baseline_contracts import comparison_identity
    from baseline_release import contract

    data = copy.deepcopy(base)
    assert len({(r["op"], r["case_id"]) for r in measured}) == len(measured)

    def reviewed_identity(op, case, params):
        # v3 publications predate the reviewed All x/input alias. Replay their
        # original joins exactly; only adaptive publications use the new join.
        if op == "AllFwdOp" and fresh_protocol == PROTOCOL:
            return op, case, contract({"workload": {"params": params}})
        return comparison_identity(op, case, params)

    fresh = {
        (
            reviewed_identity(r["op"], r["case_id"], r["params"])
            if use_v010_aliases
            else (r["op"], r["case_id"], contract({"workload": {"params": r["params"]}}))
        ): r
        for r in measured
    }
    assert len(fresh) == len(measured)
    visible = {(r["op"], r["case"]) for r in data["rows"]}
    matched = set()
    for row in data["rows"] + data.get("historical_rows", []):
        key = row["op"], row["case"]
        lookup = (
            reviewed_identity(row["op"], row["case"], row["workload"]["params"])
            if use_v010_aliases
            else (row["op"], row["case"], contract(row))
        )
        m = fresh.get(lookup)
        if m is None:
            continue
        latency = m["measurement"]["latency_ms"] * 1000
        row.update(
            tileops_latency_us=latency,
            ratio=row["latency_us"] / latency,
            tileops_verified=True,
            tileops_samples=15,
            tileops_protocol=m["measurement"].get("measurement_protocol", PROTOCOL),
            tileops_commit=commit,
            tileops_date=date,
            tileops_measurement_id=hashlib.sha256(
                json.dumps(m, sort_keys=True).encode()
            ).hexdigest(),
        )
        matched.add(((m["op"], m["case_id"]), contract({"workload": {"params": m["params"]}})))
    extra = {(r["op"], r["case"]): r for r in data.get("tileops_rows", [])}
    historical = {
        (r["op"], r["case"], contract(r)): r for r in data.get("historical_tileops_rows", [])
    }
    for m in fresh.values():
        key = m["op"], m["case_id"]
        token = contract({"workload": {"params": m["params"]}})
        if (key, token) in matched:
            continue
        tile = dict(
            op=key[0],
            case=key[1],
            workload=m["workload"],
            latency_us=m["measurement"]["latency_ms"] * 1000,
            samples=15,
            verified=True,
            commit=commit,
            date=date,
            protocol=m["measurement"].get("measurement_protocol", PROTOCOL),
        )
        if key in visible:
            historical[(*key, token)] = tile
        else:
            extra[key] = tile
    data["tileops_rows"] = [r for key, r in extra.items() if key not in visible]
    data["historical_tileops_rows"] = list(historical.values())
    data["frozen_at"] = date
    data["nightly"] = dict(
        commit=commit,
        date=date,
        fresh_cases=len(measured),
        baseline_release=base["baseline_release"],
        native_baselines="Published measurements retained with their recorded protocols",
        fresh_protocol=fresh_protocol,
        clock_policy="platform-default",
    )
    return data


def validate(path):
    path = Path(path)
    inventory = json.loads((path / "sha256.json").read_text())
    actual = {
        str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p.name != "sha256.json"
    }
    assert actual == set(inventory)
    for name, expected in inventory.items():
        p = path / name
        assert p.resolve().is_relative_to(path.resolve()) and not p.is_symlink()
        assert digest(p) == expected
    meta = json.loads((path / "meta.json").read_text())
    assert meta["schema_version"] == 2 and meta["publisher"] == "server2-physical32"
    assert meta["publisher_repository"] == "https://github.com/yyttt6/TileOPs.github.io"
    assert meta["status"] == "passed" and meta["measurement_protocol"] in (
        PROTOCOL,
        ADAPTIVE_PROTOCOL,
    )
    assert len(meta["commit"]) == 40 and all(c in "0123456789abcdef" for c in meta["commit"])
    measured = json.loads((path / "measurements.json").read_text())
    receipts = json.loads((path / "receipts.json").read_text())
    assert set(receipts) == set(meta["operators"]) == {r["op"] for r in measured}
    for op, receipt in receipts.items():
        assert {r["case_id"] for r in measured if r["op"] == op} == set(meta["case_ids"][op])
        assert receipt["guard_exit_code"] == 0
        if meta["measurement_protocol"] == ADAPTIVE_PROTOCOL:
            guard = receipt["guard_metadata"]
            assert guard["exit_code"] == 0 and guard["performance_valid"] is True
            assert guard["workload_completed"] is True
            measurement_run = receipt.get("measurement_run_id", meta["run_id"])
            if str(measurement_run) != str(meta["run_id"]):
                assert receipt.get("resumed_from"), (
                    "Prior guard needs explicit checkpoint provenance"
                )
            assert guard["owner"] == f"nightly/{measurement_run}/{receipt['batch']}"
            assert len(receipt["guard_sha256"]) == 64
        result = receipt["result"]
        assert (
            result["status"] == "passed"
            and result["failed"] == 0
            and result["source_changed"] is False
        )
        assert result["passed"] == sum(r["op"] == op for r in measured)
        assert receipt["hardware"]["cube"] == 32 and receipt["hardware"]["vector"] == 64
        assert receipt["hardware"]["clock_policy"] == "platform-default"
    assert len(measured) == meta["fresh_cases"] > 0
    for row in measured:
        assert row["raw_replayed"] is True
        assert row["status"] == "passed" and row["input_preserved"] is True
        check = row["correctness"]
        assert (
            check["checked_outputs"] == check["total_outputs"] > 0 and not check["unchecked_reason"]
        )
        m = row["measurement"]
        assert (m["warmup"], m["repeats"], m["flush_bytes"]) == (3, 15, 1 << 30)
        samples = m["latency_samples_ms"]
        assert (m["measurement_protocol"], m["profile_coverage_protocol"]) in (
            (PROTOCOL, COVERAGE),
            (SESSION_PROTOCOL, SESSION_COVERAGE),
        )
        if meta["measurement_protocol"] == PROTOCOL:
            assert m["measurement_protocol"] == PROTOCOL
        assert len(samples) == 15 and all(math.isfinite(s) and s > 0 for s in samples)
        assert math.isclose(m["latency_ms"], statistics.mean(samples), rel_tol=1e-10)
        assert m["profile_excluded_kernel_count"] == 0
        coverage = row["coverage"]
        assert len(coverage) >= 15
        assert all(len(c["complete_sha256"]) == 64 for c in coverage)
        assert all(c["result"]["profile_excluded_kernel_count"] == 0 for c in coverage[-15:])
        assert all(
            math.isclose(c["result"]["latency_ms"], s, rel_tol=1e-10)
            for c, s in zip(coverage[-15:], samples, strict=True)
        )
    base = json.loads((path / "baseline-before.json").read_text())
    version = meta.get("contract_identity_version")
    assert version in (
        None,
        "v010-reviewed-aliases-v1",
        "v010-reviewed-aliases-v2",
        "v010-reviewed-aliases-v3",
        "v010-reviewed-aliases-v4",
    )
    expected = build_cache(
        base,
        measured,
        meta["commit"],
        meta["date"],
        use_v010_aliases=version is not None,
        alias_version=version or "v010-reviewed-aliases-v1",
        fresh_protocol=meta["measurement_protocol"],
    )
    assert expected == json.loads((path / "baseline-cache.json").read_text())
    return meta, []
