# tilelang-ascend all-pairs adapters (CPU-only handoff)

`__init__.py:PROVIDERS` exports 84 lazy `factory(case) -> Candidate` entries.
All 205 operators and all 2062 rows of the saved inventory have a conclusion in
`coverage.json`. The initial assignment named 1982 cases; the shared inventory
added 80 rows while this task was running. `inventory_snapshot.json` preserves
the exact later input and its hash is recorded in coverage.

- 715 cases: `supported_unverified`, with concrete native adapters.
- 1344 cases: `unsupported` under the unchanged-original-kernel adapter scope.
- 3 TopKSelect cases: `deployment_blocked` by native whole-row UB capacity and
  the missing complete window/sentinel adapter.
- **0 device runs, 0 numerical receipts, 0 verified cases.** Historical x86
  results are not credited to this ARM64/CANN 9.1 task.

Semantic status and deployment status are separate fields. All 715 implemented
cases are also deployment-blocked: the snapshot lacks source-owned
`libtilelang.so`, `libtilelang_module.so`, `libtvm.so`, and `libtvm_runtime.so`
in the four directories inspected from `tilelang/libinfo.py`. No real runtime
import or device compilation was attempted. Original PTO headers are present;
this does not establish a working compiler deployment.

## Implementation

`plans.py` makes pure metadata capability decisions. `adapters.py` implements
layout/shape/stride/padding/dtype/API adaptations. `native.py` loads a dependency
closure of unchanged original function/class/constant AST nodes, retaining their
file and line information and excluding demo imports, cache clearing, CLI code,
random tensors and test bodies. Original schedules and pass configurations are
retained; low-level compile calls select the existing PTO backend. Original host
wrappers keep their original backend choice.

Routes include elementwise/comparison/selection, full INT64 bitwise through two
INT32 bit lanes, GEMM/BMM/grouped GEMM, im2col 1D/2D/3D convolution, max pooling,
argmin via reversed-axis native cummin, reductions/softmax, affine group/layer/
instance normalization, gated activations, rotary variants, dense/varlen/paged
GQA, native GQA forward+backward, full-key MLA, INT8 scaled GEMM/dequantization,
weighted MoE unpermute, MHC post and causal convolution prefill/decode.

Convolution bias is represented by an exact constant-one input column and the
original bias as an extra K row in the same native GEMM. Pooling unfolds windows
and delegates every maximum to the original native reduction. Argmin reverses
the scan axis to translate native last-tie cummin to first-tie argmin, then maps
the returned coordinate back. None of these routes computes the requested
reduction/matmul/activation with torch.

Input-derived transformations occur inside `run`; only compiled kernels keyed
by metadata are cached. Causal convolution operates on private state and returns
both the full output and full updated state, matching the extended harness.
GQA backward recomputes native forward output and natural-log LSE on every call;
this charges all work and avoids converting the manifest's base-2 LSE or reusing
stale forward data in q-zero controls. No oracle or tolerance was changed.

Factory entry configures build caches and temporary files under
`.cache/all-pairs/tla`, including a process-local redirect of the original
Cython adapter's cache-directory function. It adds the inspected vendor root to
the Python search path and rejects an already loaded foreign `tilelang` module.
Use the existing isolated baseline process for this library. Top-level provider
import performs none of this runtime setup and imports no torch/vendor runtime.

## CPU checks and evidence

From `TileOPs`, these commands use only Python metadata stubs and AST/source
inspection; they do not import real torch/tilelang/ACL or initialize an NPU:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m benchmarks.ascend_all_pairs.tla.static_check
PYTHONDONTWRITEBYTECODE=1 python3 -B -m benchmarks.ascend_all_pairs.tla.audit
PYTHONDONTWRITEBYTECODE=1 python3 -B -m benchmarks.ascend_all_pairs.tla.final_check
```

`static_check.py` constructs all 715 supported factories against metadata-only
modules, binds original native signatures, traverses every adapter's full output
shape/dtype flow, and checks original native write destinations against input
provenance. It also checks selected AST nodes are unchanged. This is **not a
compiler test or numerical test**. Fake native calls are never device runs.
`static_case_receipts.json` explicitly labels this limitation.

`coverage.json` includes per-op and per-case status, complete output declarations,
source file/line/SHA256, native API, adaptation, adapter path, static receipt key,
deployment status, and null numerical/ownership/input-preservation device receipt
paths. `progress.json` contains handoff counts and remaining work.
`source_catalog.json` contains 309 owned example/test/language Python files and
223 third-party catalogue entries that are explicitly excluded from attribution.

## Remaining work

A future authorized host-locked device run must establish compiler deployment,
actual numerical accuracy, full input preservation and genuine native call
ownership. None is implied by static checks. Large layout expansions and repeated
native row/window/group calls may be slow or memory-intensive; no performance
claim is made. Kernel resource usage, atomics, DMA tails, NaN/tie behavior and CANN
9.1 backend compatibility remain untested.

All 205 shared factory names have AST registration evidence at handoff. The
snapshot's older missing-factory list is retained as input history; root added
the remaining registrations during this task. Actual factory imports remain
untested under the CPU-only restriction. Opaque ADT strings in the inventory
are resolved from the matching unchanged YAML workload in `metadata.py`.

Unsupported decisions are scoped to the inspected original APIs. They do not
mean the language or hardware cannot implement these operators. Significant
mismatches include zero-row quantizer scales, missing complete LSE/state/index
outputs, feature-gating recurrence differences and fused-MoE host wrappers that
perform requested mathematics in torch. Exact reasons and source evidence are
in `audit.py` and generated coverage.
