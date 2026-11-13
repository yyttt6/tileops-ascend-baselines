# Closing delivery status

The independent reviewer confirmed **475/475 known closing-queue contracts and 14,250 raw windows** on 2026-11-13 ([review](https://github.com/yyttt6/agent-hub/issues/3#issuecomment-6852766311)). There are **474 complete-output candidates / 14,220 windows**, one excluded partial-output candidate (DeltaNetChunkFwdOp), and **zero final acceptances**. These are cases/contracts, not 475 different operators. This is not completion of the entire 2,267-contract inventory.

## Evidence index

Detailed records remain in the existing private [evidence release](https://github.com/yyttt6/agent-hub/releases/tag/evidence-torch-compile-v2-20261111), asset `closing-delivery-index01.tar.gz` (1,125,078 bytes, SHA256 `fe2b89d984ff4237fea2043b26e882f22d66d36689142941fa76684548e71308`). No raw traces, environment fingerprints, tensors, binaries or generated reports are added to this PR.

The index contains:

- `contracts.json`: 475 unique contract SHA/op/case entries, environment/protocol labels, complete-output/exclusion flags, archive and nested member identity, admission SHA, mean latency, and all 14,250 variant/window indices with latency and trace SHA. Historical duplicate/pilot/failure admissions are retained as attempts.
- `archives.json`: 79 uploaded archive names, byte sizes, SHA256 and private release URLs. Nested members use `outer.tar::member` notation.
- `original39.json`: the original 48 contracts retain 44 original-environment successes, one original Round compile failure and three original-environment unattempted cases. Four patched-environment successes are separate, with their audit reference.
- `inventory.json`: all 2,267 unique inventory contracts, separate prior name matches and archived attempts.
- `summary.json` and `MANIFEST.json`: counts, unresolved alignment limits and index-file hashes.

The index is a consolidation of existing admissions and the cited independent review, not a new raw replay. Historical admission protocol strings remain in the raw archives; resident/patched labels in the index distinguish later execution. Original environment details must be read from each archive; a shared version string does not establish environment equivalence.

## Inventory reconciliation still open

Current archive-backed partition: 475 newly measured/audited contracts, 70 with archived attempts but no complete measurement in this index, 357 with scope pending exact-contract review (not excluded), and 1,365 with no device attempt found in the indexed archives. These sum to 2,267. The 70 include diagnostic/admission/guard failures and must not all be called compile failures. No indexed attempt does not imply that a case is runnable or has never been tested elsewhere.

CPU reconciliation now resolves the 67 historical name-matched contracts into 83 records: 41 quant records have exactly matching frozen parameters, 29 are one-window pilots, seven are compile failures, and six are historical 15-window formal/raw-verified records. Records overlap on contract identity, so these numbers must not be added as unique coverage. Seven other historical records fall outside this frozen inventory. Source/environment equivalence remains unverified; none is silently promoted to same-protocol coverage.

All 2,267 contracts (206 op names) have corresponding name declarations in frozen benchmark factory modules. This static check does not execute constructors, allocate inputs, or prove a usable runtime reference. It finds no basis for marking any inventory row “no implementation”; that status remains unproven rather than fabricated. Runtime callable checks, where needed, remain separate from this metadata reconciliation.

Reproduce with `python3 tools/resident_measurement/reconcile_inventory.py --checkout . --output /path/to/new-inventory.json`. The per-contract result is private release asset `inventory-reconciliation02.tar.gz`, 126,408 bytes, SHA256 `db743a2df4ab902472ab164f7140fddc810d5d12dde7a1e4df894bd3d978ed4d`. It supplements the first index without rewriting its historical summary. It records source filenames, line numbers, source hashes, prior record classifications and explicit unverified boundaries. No torch import or NPU operation is performed.

## Acceptance limits

Original wheel origin is not fully recovered. Environment version/path/hash checks and isolated copying do not reconstruct a reproducible wheel supply chain. The measured patched environment has backend SHA `0be87418…`; its four results cannot replace original-environment failures.

Device ownership observation is container driver-based plus cooperative host reservation. Full process FD visibility was deliberately not required by the user-approved protocol. Discrete sampling cannot prove continuous absence of unrelated activity. Do not reopen permissions or weaken the checker to change this boundary. Historical guard 78/75 attempts remain failures even when later attempts succeeded.

Runtime and flush compilation use different CANN profiles. The historical CPU rebuild recipe targets 9aebeb… and cannot directly satisfy the frozen runner requirement 693be2fb78b6c749ec2afee81e1ed3d8c0de58ea3397daaeaeef06588cc2e865. Execution requires the already audited matching ARM64 aibin and compatible host-loader assets/provenance; this checkout does not provide a standalone verified build of those assets. Check each archive's actual flush SHA/compiler receipt. Protocol variants must not be pooled as an identical environment.

Diagnostic02 failed before compile; diagnostic03 reproduced the original compile error and passed the patched float case plus 18 float regressions, then failed on an unsupported CPU integer reference with five remaining tests unattempted. This is not 24 passing regressions. The later four Round performance cases passed independently. Historical CDLL root cause remains unresolved; the independently validated resident sampler is the measured workaround.

Cross-case CPU compilation overlap remains experimental. The prior device A/B showed wall-time savings but not timing equivalence; the closing measurements remain serial. Full-output exclusion, environment acceptance and final publication are still reviewer decisions. This Draft PR does not merge or mark the commission complete.

## CPU delivery validation

A clean export from the staged Git tree passed all 11 CPU tests, reconstructed and verified 589 frozen source files plus two pinned measurement tools, and materialized the exact device-tested backend SHA from the retained original source. No NPU work was launched. The backend materialization positive test uses the privately retained original file; that wheel/source prerequisite is not distributed by this PR. Existing trailing whitespace in the byte-pinned measurement runner and historical unified patches is preserved intentionally; plain `git diff --check` reports those lines. No device evidence is claimed for changed helper code.

A subsequent clean-directory import check found that source reconstruction omitted `baseline_protocol.py`, which the measurement core imports. The helper now copies and hash-checks that dependency too (589 source files plus three tools), with a regression test that imports the reconstructed measurement core and full32 parser under `python -S` without importing torch or torch_npu. This fixes the reproduction helper only; the byte-pinned device runner and historical manifest remain unchanged.

## Scope correction after independent CPU review

The 357 rows formerly labelled `outside_current_specialist_scope_provisional` in the immutable first index are now **scope pending**, not excluded. That old op-level list includes ordinary Add/Mul/normalization/pool/softmax contracts. Only exact-contract evidence of existing coverage, a specifically excluded quant/anomaly-attribution task, or an explicit receiving owner can settle an individual row. An op name alone is insufficient. No recipient assignment or permanent exclusion is inferred here. The old archive is retained; a separate per-contract correction ledger records the current status. The other 1,365 unattempted-in-index rows also do not imply completion.

Correction asset: `scope-correction03.tar.gz` in the same private release, 20,339 bytes, SHA256 `84fce4ac82eb75056e6822871c4617c3ab57341e59aebaa25cb31fe0b43efc9a`. It contains 357 unique contract identities, old/current categories, explicit non-exclusion and unverified ownership flags, and required resolution evidence. Reproduce with `reclassify_scope.py --inventory /path/to/original/inventory.json --output /path/to/new-correction.json`; the source inventory is hash-bound and never rewritten.
