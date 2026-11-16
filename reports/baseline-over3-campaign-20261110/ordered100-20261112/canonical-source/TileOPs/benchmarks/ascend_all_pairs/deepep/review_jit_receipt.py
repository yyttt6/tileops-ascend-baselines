"""Offline stdlib-only review of a DeepEP JIT provenance receipt on its host.

This checks evidence consistency and recorded/current hashes, not actual ACL
execution. It never promotes an official-log receipt to native-pass eligibility.
Usage: python -I -S review_jit_receipt.py --workspace ROOT --row checker.json
       --final /path/to/jit-evidence-*/final.json
"""

import argparse
from pathlib import Path

if __package__:
    from .jit_provenance import (JITProvenance, REQUIRED, SCHEMA, file_record,
                                 parse_log, read_json, require)
else:
    # Isolated (-I -S) Python omits the script directory from sys.path.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from jit_provenance import (JITProvenance, REQUIRED, SCHEMA, file_record,
                                    parse_log, read_json, require)
    finally:
        sys.path.pop(0)


def review(workspace, row, final_path):
    workspace, final_path = Path(workspace).resolve(), Path(final_path).resolve()
    final = read_json(final_path)
    sources = row["source_sha256"]

    def checked_file(path, recorded=None):
        current = file_record(path)
        path = Path(current["path"])
        key = str(path.relative_to(workspace)) if path.is_relative_to(workspace) else str(path)
        require(sources.get(key) == current["sha256"], f"missing/mismatching Candidate source hash: {key}")
        if recorded is not None:
            require(current == recorded, f"recorded artifact replaced since observation: {path}")
        return current

    checked_file(final_path)
    require(final["schema"] == SCHEMA and final["status"] == "load_provenance_collected",
            "final receipt is not a completed load-provenance collection")
    require(final["native_pass"] is False and final["direct_acl_launch_attestation"] is False
            and final["loaded_byte_identity_attested"] is False, "unsupported native/loaded-byte claim")
    require(row["op"] == final["case"]["op"] and row["case_id"] == final["case"]["case_id"],
            "checker row and JIT receipt identify different cases")
    require(row.get("library") == "DeepEP-Ascend", "checker row is not DeepEP-Ascend")
    # This is intentionally a live-host review: inode/time pins reject replaced
    # artifacts. Archive/cross-host copies require a distinct content-only audit.
    start_path = final_path.with_name("start.json")
    checked_file(start_path)
    start = read_json(start_path)
    require(start["case"] == final["case"] and start["pid"] == final["pid"]
            and start["initial_cache_empty"] is True and start["cache_root"] == final["cache_root"],
            "start/final ownership or case mismatch")
    for record in start["source_files_before"].values():
        checked_file(record["path"], record)
    observer = object.__new__(JITProvenance)  # No constructor, device, capture or environment side effects.
    observer.root = Path(start["cache_root"])
    observer.vendor, observer.build = Path(start["vendor_root"]), Path(start["build_root"])
    observer.tools = {k: Path(v) for k, v in start["toolkit"].items()}
    observer.pinned = start["source_files_before"]
    seen, run_entries, runs, offset = {}, set(), 0, 0
    for index, scope_path in enumerate(final["scopes"]):
        scope_path = Path(scope_path)
        require(scope_path == final_path.parent / f"scope-{index:04d}.json", "foreign/reordered scope path")
        checked_file(scope_path)
        scope = read_json(scope_path)
        require(scope["schema"] == SCHEMA and scope["case"] == final["case"]
                and scope["pid"] == final["pid"] and scope["index"] == index,
                "scope identity/sequence mismatch")
        log_path = scope_path.with_suffix(".log")
        require(scope["stdout"]["path"] == str(log_path), "foreign scope log")
        checked_file(log_path, scope["stdout"])
        data = log_path.read_bytes()
        require(scope["stdout_byte_range"] == [offset, offset + len(data)], "non-contiguous capture offsets")
        offset += len(data)
        text = data.decode("utf-8")
        loads, commands = parse_log(text)
        require(loads == scope["successful_loads"] and set(loads) == set(scope["new_entries"]),
                "scope does not associate each new object with successful load diagnostics")
        require(scope["phase"] in ("initialization", "factory_preparation", "before_run", "candidate_run", "cleanup")
                and (scope["phase"] != "before_run" or not loads), "invalid/unscoped JIT activity")
        require(scope["known_entries_stable"] == sorted(seen), "scope previous-object set mismatch")
        associations = {}
        for path, entry in scope["new_entries"].items():
            require(path not in seen and Path(path).parent == observer.root / "cache",
                    "reused/foreign entry in receipt")
            for record in entry["files"].values():
                checked_file(record["path"], record)
            associations[path] = observer._metadata(path, entry, commands, text.splitlines())
            seen[path] = {"snapshot": entry, "first_scope": index}
        require(associations == scope["associations"], "source/command/ELF association does not reproduce")
        require(sorted(commands) == sorted(c for a in associations.values() for c in a["observed_commands"]),
                "unattributed compilation in scope")
        if scope["phase"] == "candidate_run":
            require(scope["run_returned_and_synchronized"] is True, "candidate run did not complete/synchronize")
            if runs == 0:
                require(REQUIRED[final["case"]["op"]] <= {v["tag"] for v in scope["new_entries"].values()},
                        "first run lacks required case loads")
            runs += 1
            require(scope["run_index"] == runs, "candidate run sequence mismatch")
            run_entries.update(loads)
        else:
            require(scope["run_returned_and_synchronized"] is False and scope["run_index"] is None,
                    "non-run phase falsely claims a case run")
    require(runs > 0 and runs == final["runs"] and seen == final["entries"]
            and sorted(run_entries) == final["candidate_run_entries"], "final aggregate mismatch")
    require(observer._cache() == {k: v["snapshot"] for k, v in seen.items()}, "live cache differs from final receipt")
    raw_path = final_path.with_name("stdout.raw.log")
    require(final["stdout"]["path"] == str(raw_path), "foreign raw capture")
    checked_file(raw_path, final["stdout"])
    require(raw_path.read_bytes() == b"".join(Path(p).with_suffix(".log").read_bytes() for p in final["scopes"]),
            "raw stdout and scoped logs differ")
    return {"schema": "deepep-jit-receipt-review/v1", "case": final["case"],
            "status": "load_receipt_consistent", "objects": len(seen), "runs": runs,
            "native_pass": False, "native_admission": "requires_device_launch_and_loaded_bytes_evidence",
            "checker_status_observed": row.get("status"),
            "device_execution_performed_by_reviewer": False}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--row", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.workspace, read_json(args.row), args.final), indent=2))


if __name__ == "__main__":
    main()
