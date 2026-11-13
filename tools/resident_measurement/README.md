# Resident measurement delivery

This directory preserves the reviewed four-contract Round controller and the resident driver sampler used for the closing measurements. The 23-file `MANIFEST.json` is unchanged from device testing (SHA256 `deafc7c57a9bdd0968bd9b1ed5465a36a1a3b2bc748a96a9561beeb1188f113b`). Its `PLAN.txt`, legacy status strings and patches are historical preparation records, not current acceptance claims. Additional delivery helpers are outside that historical manifest and versioned by Git.

`resident.py`, `integration.py`, `watchdog.py`, and `candidate_guard.py` supply the reusable sampler/lifecycle. `measure_driver_provenance_v2.py` accepts a frozen claim and index. The bundled controller intentionally restricts execution to four reviewed contracts, stops on any failure, and never retries. Adapting another queue requires an explicit claim/manifest and environment binding. This is serial NPU measurement; the separate experimental `compile_pipeline` is not enabled here.

## CPU validation from a clean checkout

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/resident_measurement -p 'test_*.py'
python3 tools/resident_measurement/prepare_sources.py --checkout "$PWD" --destination "$RUN_ROOT/frozen"
```

Use a new absolute `RUN_ROOT`. Source preparation verifies 589 frozen files plus the pinned measurement core, baseline protocol helper and full32 parser from files already tracked in this repository. It does not import torch or touch the device. Tests check reviewed hashes, entry points without site packages, fail-stop queue behavior and real CPU child launch/timeout/reaping.

## Isolated backend preparation

Provide the compatible original environment at `$RUN_ROOT/venv`. Original wheel provenance remains incomplete; this is a prerequisite, not a claimed reproducible wheel installation. Do not substitute a different torch build based only on version strings.

```sh
python3 tools/resident_measurement/materialize_patch.py \
  --source "$RUN_ROOT/venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/post_grad.py" \
  --output "$RUN_ROOT/post_grad.candidate.py"
mkdir "$RUN_ROOT/round-isolated-envs"
python3 tools/resident_measurement/prepare_envs.py \
  --source "$RUN_ROOT/venv" --destination-parent "$RUN_ROOT/round-isolated-envs" \
  --candidate "$RUN_ROOT/post_grad.candidate.py" \
  --candidate-sha 0be87418b543e882f4e7351a21a0a9bcf578b33a8be271c81b0e861d6da853db
```

The patch materializer rejects any original backend except SHA256 `a1779017efc75841fc03a9561f902ec2a25e8a6de709356f1a9700a077ec803a`, and verifies the candidate SHA before creating a new file. The environment preparer copies independent files, refuses existing/overlapping destinations, and records every file hash. Shared site-packages are never edited. The guarded entry checks actual interpreter, module paths, dependencies and all environment hashes before measurement. The measured environment contained 70,867 files; a new environment must undergo its own provenance review.

## Measurement command and prerequisites

The command below requires **already audited matching ARM64 runtime assets** in `$RUN_ROOT/flush-rebuild/build01`: `flush.aibin` with SHA256 `693be2fb78b6c749ec2afee81e1ed3d8c0de58ea3397daaeaeef06588cc2e865` and its audited compatible host loader `flush.so`. Obtain their associated source/compiler/build and host-loader provenance receipts through the private evidence channel before attempting reproduction. A matching aibin hash alone does not validate an arbitrary loader. The source checkout does not supply these binaries or a complete standalone recipe proven to reproduce them.

**The historical `commissions/torch-compile-20261110/flush_source/CPU_REBUILD.md` recipe targets `9aebeb…`, not the required `693be2…`. It cannot be used as a direct substitute for this runner's assets.** Do not relax the frozen SHA assertion to make a new build run. If matching audited assets and provenance are unavailable, the measurement command is not ready to execute; CPU tests and source reconstruction remain available.

Keep the CANN 9.2 ARM64 flush compiler profile separate from the CANN 9.1 runtime profile. The measured runtime used Python 3.12.15, torch 2.13.0+cpu, torch_npu 2.13.0.rc1, AscendC, 32 AIC / 64 AIV. Source the site-provided compatible runtime activation script before the command below; no credentials or activation paths are embedded here.

The host reservation `/run/lock/ascend-npu/device-0.lock` and the sampler's device mapping must match the reviewed host setup. Do not run on another device mapping without review. A held lock alone does not prove exclusive device use: the resident driver sampler also validates ownership/health throughout measurement.

```sh
export PYTHONDONTWRITEBYTECODE=1
python3 tools/resident_measurement/controller.py \
  --root "$RUN_ROOT" --output "$RUN_ROOT/round-measurement04-run" \
  --manifest-sha deafc7c57a9bdd0968bd9b1ed5465a36a1a3b2bc748a96a9561beeb1188f113b
```

The output must be new. The unchanged protocol uses three warmups, 15 eager and 15 compile windows, 1 GiB flush, full32+AICPU parsing, original complete-output checks, zero-output rejection and graph counters. Sampler query deadline is 10 seconds; admission wait and foreground budgets are each 600 seconds. Failure stops the queue and retains evidence. The command is documentation, not a request to rerun completed cases.

The final four device results were independently audited in agent-hub issue #3 comment 6852766311. See `DELIVERY_STATUS.md` for the closing queue, private evidence index and unresolved acceptance boundaries.
