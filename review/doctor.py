"""CPU-only checkout verification, without importing Torch or NPU."""
import argparse
import ast
from review.common import HERE, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-edited-baselines", action="store_true")
    args = parser.parse_args()
    export = read(HERE / "catalog/EXPORT.json")
    status = read(HERE / "catalog/CASE_STATUS.json")
    accepted = read(HERE / "catalog/ACCEPTED_SELECTIONS.json")
    assert len(status["rows"]) == 665
    assert sum(r["verified_within3"] for r in status["rows"]) == len(accepted["cases"]) == 178
    assert len({(r["op"], r["case"]) for r in status["rows"]}) == 665
    for relative, record in export["contract_and_measurement_sources"].items():
        assert sha(HERE / relative) == record["sha256"], relative
    edited = []
    for name, digest in export["selected_source_files"].items():
        if sha(HERE / "baselines" / name) != digest:
            edited.append(name)
    if edited and not args.allow_edited_baselines:
        raise ValueError(f"Edited baseline source (expected during development):{edited}; use --allow-edited-baselines")
    for name, digest in export.get("additional_pending_route_sources", {}).items():
        if sha(HERE / "baselines" / name) != digest and not args.allow_edited_baselines:
            raise ValueError(f"Edited pending route:{name}")
    for row in accepted["cases"]:
        assert row["ratio"] <= 3 and row["verified_within3"]
        if row["ratio"] >= 2.9:
            assert row["candidate_processes"] >= 2
        name, symbol = row["candidate_entry"].split(":")
        tree = ast.parse((HERE / "baselines" / name).read_text())
        assert any(isinstance(n, ast.FunctionDef) and n.name == symbol for n in tree.body)
    print("PASS:665 unique cases;178 historical accepted selections; original source SHA; selected entries")
    print("No device operation performed. New portable runner still requires local machine validation.")
    if edited:
        print("Development changes require new measurements; historical results remain bound to original hashes:", edited)


if __name__ == "__main__":
    main()
