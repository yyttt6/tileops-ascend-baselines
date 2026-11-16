# TileKernels all-pairs providers

Import `benchmarks.ascend_all_pairs.tilekernels.PROVIDERS` for exact-op-name → `Case -> Candidate` factories. The package entry point imports no Torch, NPU runtime, TileLang or vendor package. Factories return only original TileKernels native calls with layout, padding, exact input widening, output selection and dtype restoration. No vendor source or shared registry is modified.

Adapters keep all input-dependent packing inside `run`; original API calls, including unused fused outputs, are part of execution cost. Source/API/adaptation fingerprints are attached to Candidates. FP16 fused residual RMSNorm preserves the manifest's residual rounding between two original calls. FP8 per-group quantization embeds the complete group into a separate native32×32 block; it does not merge or split scale reductions. Large native TopK/RMS reductions remain blocked by original UB limits.

`check.py` reuses the unchanged shared `ascend_classic_check.py`, including full-output verification, input preservation and Add/FP8/RoPE special checks. Its isolated in-process factory mapping does not mutate the shared registry file. Receipts add original vendor Python stacks and native TileLang/Ascend launch names/generated-source/binary hashes. It refuses execution outside `npu-guard` before any runtime imports.

From the workspace root, with a fresh output path:

```bash
npu-guard run --owner allpairs/tilekernels --wait 3600 --timeout 300 -- \
  .tools/baselines/baseline-run TileKernels python -B \
  TileOPs/benchmarks/ascend_all_pairs/tilekernels/check.py \
  --op SiluAndMulFwdOp --case-id llama-8b-ffn-decode-bfloat16 \
  --library TileKernels \
  --output .agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilekernels/receipts/manual-new.json
```

`batch.py QUEUE_JSON [BUDGET_SECONDS]` runs exact cases sequentially, skips existing receipts and stops before starting a case beyond the budget. Use245 seconds with a300-second guard, or the default780-second budget within a900-second guard only when compilation warrants it. Do not modify active adapters while a batch runs.

`report.py` is CPU-only and rebuilds the task-local `coverage.json`, `progress.json` and source inventory. Coverage separates mathematical absence, unsupported input dtypes, native resource/deployment failures, numerical failures and implemented-but-unverified cases. Historical passes are accepted only with the recorded source fingerprints still matching. All205 operators and2062 full-manifest cases are accounted for; missing shared workload factories are explicit. The inventory copy and immutable raw receipts remain in the task evidence directory. Compiler/cache/temp files are isolated under `.cache/all-pairs/tilekernels`.

`DIAGNOSTIC_PROVIDERS` retains the native FusedQKNormRoPE composition for reproducing its five numerical failures. It is excluded from the formal `PROVIDERS` registry. Per-case numerical failures in an otherwise useful provider remain explicit in coverage, so consumers must select verified cases before timing.

Composed providers enumerate their original APIs and execute every required native call. Some materialize large padded buffers or unused fused outputs. This task records correctness and provenance; it does not claim a speedup. Maximum-pool windows preserve their complete reduction and source-index order; padding duplicates the first valid source index of that window. Average-pool padding/divisors and ragged chunk metadata retain the manifest contract.

`pooling_max.py` selects original TopK `k=1` over each complete window. Ragged windows duplicate their first valid source index; the packed-input synchronization is included in execution cost. It does not modify the original kernel or split a reduction. Earlier device exceptions were independently reproduced by the original NPU adaptive-maxpool reference before any vendor call; those receipts cannot establish a TileKernels kernel fault. `original_pooling_oracle_diagnostic.json` records that proof and the successful CPU execution of the same reference function. The isolated checker uses the existing `cpu_reference` helper for maximum-pool checks and explicitly records placement/source hashes. Manifest/oracle source and exact value/index tolerance remain unchanged.

A batch process exiting zero does not mean every case passed, nor that every queued case ran: the runner records individual outcomes and stops on device synchronization/cleanup failure or its time budget. Inspect raw receipts and regenerate coverage. The evidence directory preserves every earlier attempt with distinct receipt names.

After device commands finish, run these CPU-only checks from the workspace root:

```bash
python3 -B TileOPs/benchmarks/ascend_all_pairs/tilekernels/report.py
python3 -B TileOPs/benchmarks/ascend_all_pairs/tilekernels/audit.py
```

The artifact audit follows the currently accepted receipt for each case, checks current Candidate/dependency fingerprints, and matches actual native launches to isolated compiled artifacts. Superseded attempts remain archived in the same evidence directory. Trace version2 reads cached generated sources as well as loaded binary hashes, so cached kernel signatures remain distinguishable.

Final validation (2026-10-21):205 operators /2062 manifest cases reviewed;76 formal providers, all with device passes.545 cases verified (504 fresh,41 historical with matching source hashes);119 native compiled variants audited with0 problems. Remaining classifications:1474 native-contract mismatches/absence,25 deployment/resource limits,12 unsupported input dtypes,6 numerical failures (Sum1,FusedQKNormRoPE5). No pending cases or source-attribution reviews remain.42 MaxPool cases retain the explicit same-oracle CPU placement described above. Detailed statuses and immutable evidence are in the task-local coverage.json, native_artifact_audit.json, final_integrity.json and ORACLE_ISSUE.md.
