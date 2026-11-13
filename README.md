# TileOPs Ascend Baselines

Ascend baseline adapters, benchmark support code, pinned upstream dependencies,
and compact result catalogs collected for TileOPs, including the fixed NV
TileOPs v0.1.0 retest campaigns. This repository is a source and result snapshot,
not a claim that every included adapter has passed the current contract.

## 2026-11-13 baseline fixes

See [handoff](HANDOFF-20261113.txt): 28 corrected and remeasured torch pairs,
132-case metadata matching patches, and the exact CI snapshot/SHA.
Raw traces remain private; upstream CI/site integration is separate.

## Layout

- `TileOPs/benchmarks/`: public provider registry, library adapters, case definitions and timing support.
- `TileOPs/workloads/`, `TileOPs/src/`, `TileOPs/tests/`: the corresponding compatibility/source snapshot; local modifications are preserved.
- `ascend_baselines/`: upstream libraries as pinned Git submodules. See `dependencies.lock.json` and `patches/` for local tracked source changes.
- `reports/`: campaign-specific adapter/runner **source snapshots**, including earlier versions. They are not live controllers.
- `reports/torch-v010-20261106/source*`: canonical source snapshots used by the torch campaign; do not mix them with the older working-tree contract.
- `tools/`: measurement, verification and inspection source.
- `catalog/results.json`: current native/torch campaign status and exact case contracts.
- `catalog/campaigns.json`: historical campaign directories included in the source export.
- `catalog/source-manifest.json`: SHA256 and origin of exported source files.

## Quant/dequant correction (2026-11-10)

See [baseline adjustment methods (中文)](docs/BASELINE_ADJUSTMENT_METHODS.zh.md) for the applied techniques, validated examples, and limits.

The dedicated server2 campaign covers all 71 fixed workloads with same-campaign eager controls. It adds 89 verified non-control variants, including explicitly separated Torch-expression compilation and compiled native antiquant wrappers. Exact cases, callable factories, source pins, and limitations are in [the campaign directory](reports/quant-baseline-fix-server2-20261110/USAGE.json) and [report](reports/quant-baseline-fix-server2-20261110/REPORT.json). Failed and unmeasured candidates remain separate in the catalog. These are device measurements; no raw data or website update is included. The general baseline retest remains paused.

## Status snapshot

At the explicit pause on 2026-11-09:

| Campaign | Accepted | Attempted, not accepted | Unmeasured |
| --- | ---: | ---: | ---: |
| Native flex batches 1–53 | 956 | 34 | 14 (7 automatically deferred) |
| Fixed v0.1.0 torch batch | 234 | 0 | 0 |

Counts are library × operator × case pairs, not operator counts. These campaigns
are not the whole project denominator. The native review union contains 2278
pairs; 1274 have not been assigned to the flex campaign. Most of the union also
belongs to the historical v3 backlog; do not add those scopes together.

**Retesting is paused.** This export does not start controllers, install timers,
or schedule NPU jobs. Old source comments/commands describe historical runs and
are not current authorization to execute them.

## Checkout and use

```sh
git clone https://github.com/yyttt6/tileops-ascend-baselines.git
cd tileops-ascend-baselines
python3 scripts/catalog.py summary
python3 scripts/verify_sources.py
# Optional, downloads upstream sources only:
git submodule update --init --recursive
```

Source hashes describe this export, not a new scientific validation. Runtime
scripts contain original machine paths and library build requirements. Read
[portability and measurement](docs/REPRODUCIBILITY.md) before configuring a new
machine. Upstream library revisions do not by themselves reproduce a compiled
ARM64/CANN artifact. No NPU test was run for this repository export.

## Data deliberately excluded

No profiler traces, full-receipt archives, cold-storage backups, tensors, compiled
libraries, runtime caches, credentials, agent memory, or live service state are
uploaded. `raw_sha256` identifies retained private evidence where available; it
is not a download URL. Full evidence stays at its original location. The public
catalog alone cannot independently replay the original profiling windows.

## Licensing

Upstream licenses remain applicable; TileOPs' license is retained under
`TileOPs/LICENSE`, and dependency licenses are available in their pinned source
repositories. This collection does not relicense third-party code. Historical
source snapshots retain existing notices. See [dependency provenance](dependencies.lock.json).

## External torch.compile commission

- [Ascend 950 torch.compile 补测任务书（无 Docker、环境分层、分批交付）](commissions/torch-compile-20261110/README.zh.md)
