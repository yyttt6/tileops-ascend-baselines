"""CPU-only full15 raw task decomposition of one local review result."""
import argparse
import collections
import importlib.util
from pathlib import Path
from review.common import HERE, read, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    row = read(a.result)
    spec = importlib.util.spec_from_file_location("full32", HERE / "support/tools/kernel_task_profile_full32.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    means, counts = collections.defaultdict(float), collections.Counter()
    profiles = row["timing"]["retained_profiles"]
    if len(profiles) != 15:
        raise ValueError("Require all15 formal windows")
    for saved in profiles:
        path = Path(saved["profile_path"])
        if parser.trace_digest(path) != saved["profile_trace_sha256"]:
            raise ValueError("Trace identity changed")
        window = parser.parse_profile(path, 1)
        if not window["profile_named_task_coverage_complete"]:
            raise ValueError("Incomplete task coverage")
        for task in window["profile_kernel_tasks"]:
            key = (task["name"], ",".join(task["core_roles"]))
            means[key] += task["duration_us"] / 15
            counts[key] += 1
    tasks = [{"task": k[0], "roles": k[1], "mean_us": v,
              "mean_calls": counts[k] / 15, "fraction": v / row["device_us"]}
             for k, v in sorted(means.items(), key=lambda item: -item[1])]
    write_new(a.output, {"key": row["key"], "device_us": row["device_us"], "tasks": tasks,
        "interpretation": "Observed device components, not proof of an optimization ceiling or reason to stop."})
    for task in tasks[:20]:
        print(f"{task['mean_us']:12.4f} us {task['fraction']:7.2%} {task['roles']:8} {task['task']}")


if __name__ == "__main__":
    main()
