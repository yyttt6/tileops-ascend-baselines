# Ascend 950 baseline collection for 12 additional TileOPs interfaces

The inventory contains 100 manifest cases. Each entry binds the case ID to its constructor parameters, tensor shapes and dtypes at the pinned upstream revision. The existing 206 Ascend interfaces have a separate release history.

Use the TileOPs checkout named in `cases.json` with the project's Ascend 950 Python environment. Select a case with `benchmarks.api.cases(op)`. Its `case.torch` gives the upstream eager implementation and `case.reference` gives the workload oracle. Some workloads intentionally use the same function for both. For mutable inputs, clone and restore the private buffers before each repeat, outside the timed region.

The measurement entry point uses the Ascend CI timer, including three warmups, fifteen repeats and a 1 GiB cache flush per repeat. The last timed output goes through the workload's numerical checks. The resulting per-case JSON and raw profiler paths are device evidence. Each run also needs an uncontested host-guard receipt and its frozen source identity before release. The inventory alone provides no measured latency.

To reconstruct the actual measurement source from the Ascend base, check out `b47b02deea178d662d31c51d5b723b6739a465f4` and apply `gzip -dc frozen-source.patch.gz | git apply --index`. The resulting Git tree is `b9a51c049d90518788ac2cd551cc900beb900f6b`. Run `measure_reference.py OP OUTPUT` under the shared Ascend device guard with this checkout, its `src`, `scripts/ci` and `scripts/ci/nightly_support` on the Python path. `torch-eager.json` lists the measured case IDs, the 15 replayed sample values in microseconds and each raw profile digest.
