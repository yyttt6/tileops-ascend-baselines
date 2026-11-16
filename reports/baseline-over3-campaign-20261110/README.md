# Frozen baseline refresh, 2026-11-16

This directory preserves code and qualified result summaries from the baseline >3x review campaign. Start at `../../catalog/baseline-over3-refresh-20261116/summary.json` and `cases.json`. All 2119 current-scope cases have qualified coverage; this does not mean all final speedup ratios are closed or that anything has been integrated into CI/nightly.

`ordered100-20261112/RESULT.json` contains the explicit first100 selection. The original continuation ledger and named `RESOLVED_*.json` overlays contain the remaining qualified routes. The derived catalog points to the exact JSON records and keeps target library keys separate from alternative providers and evidence reuse. It does not select the smallest latency across different runtimes.

`ordered100-20261112/canonical-source/TileOPs` is the actual server1 frozen canonical source snapshot. It is separate from the older root `TileOPs` tree. Other Python/shell/native text files here are the original task adapters, support, validation and historical controller sources, often with machine-specific `/opt/...` paths. Do not import queue/worker/controller scripts as a library or treat snapshots as run authorization.

Tensors, compiled binaries, raw profiles and large archives are excluded. Recorded original paths and SHA256 values identify retained evidence; they are not public download URLs. SDK AICPU declarations and retained private AICORE ELF proofs have different qualification boundaries. The code/source snapshot does not install or rebuild private native packages.
