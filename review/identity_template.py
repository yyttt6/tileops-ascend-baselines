"""Create an incomplete review template bound to exact local result files."""
import argparse
from review.common import sha, write_new
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", nargs="+", type=Path, required=True)
    p.add_argument("--tileops", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    write_new(a.output, {"reviewer": "", "physical_device_identity": "",
        "same_physical_device_driver_firmware": False,
        "approved_runtime_profiles": False,
        "executing_native_and_jit_binary_identity": False,
        "full_api_and_unchanged_precision": False,
        "no_interference_or_profiler_loss": False,
        "notes_and_evidence_paths": {},
        "result_sha256": {str(p.resolve()): sha(p) for p in [*a.baseline, a.tileops]}})
    print("Incomplete template written. Fill only after checking and recording the actual evidence.")


if __name__ == "__main__":
    main()
