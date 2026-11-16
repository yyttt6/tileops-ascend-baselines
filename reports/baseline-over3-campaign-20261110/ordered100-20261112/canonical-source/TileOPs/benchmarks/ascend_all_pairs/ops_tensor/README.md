# ops-tensor all-pairs adapter

The import-safe endpoint is `benchmarks.ascend_all_pairs.ops_tensor:PROVIDERS`.
Each value accepts the original `Case` and returns `benchmarks.ascend_baselines.Candidate`.
Torch, native loading and compiler calls are inside factories. Importing the endpoint
uses only the standard library and the local metadata/adapter modules.

This worker was strictly CPU-only. **No NPU initialization, binary loading, numerical
test, benchmark or profiling occurred.** `supported_unverified` means a concrete
adapter and matching source/metadata route with a successful native compile. It does
not mean the output, input protection or runtime launch attribution passed.

The shared inventory changed during the task from 1982 factory-backed cases to
2062 manifest cases, still covering 205 operators. The captured inventory is
`.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_tensor/inventory_snapshot.json`.
`coverage.json` retains `in_original_1982_scope` on every case and both totals.

| Route | Registered operators | Current admitted cases |
| --- | --- | ---: |
| Original 64-element Add | Add | 12 |
| Original 256-element Trunc | Trunc | 5 |
| Original full GEMM; batch/group slicing | Gemm, Bmm, GroupedGemm, MoEGroupedGemm | 52 |
| Sliding-window layout plus original full GEMM/native bias | Conv1d, Conv2d, Conv3d | 55 |
| Chunk/group GEMM and causal output crop | SSDChunkCoupling | 3 |
| Original 64-lane max reduction; whole pooling windows | MaxPool1d/2d/3d, AdaptiveMaxPool2d | 18 |
| Original FP32 GEMM with constant one/reciprocal coefficients | Sum, Mean, AvgPool1d/2d/3d, AdaptiveAvgPool2d, MeanPooling | 37 |

There are **21 registered operators, 182 supported_unverified cases, and zero verified
cases**. The remaining 1800 cases of the original 1982 scope are classified with
source evidence. The added 80 cases are also screened; at the final source audit,
35 belonged to six still-missing workload factories, recorded independently from
mathematical support. Factory availability is checked against the live shared
registry, which continued to grow after the inventory snapshot was captured.

All four bridges include the complete vendor source and call its existing launcher.
They define no new device kernel. GEMM uses the original `GetTilingConfig`,
`ComputeTiling`, `LaunchKernel` and `GemmUniversal`; no tiling, scheduling, math or
tolerance edits were made. Its original FP32 option disables HF32. Half/BF16 input
widening is exact. All input-dependent packing and metadata reads happen in `run`.
Output allocations are private. Constant-operand reductions still need device
validation for accumulation order and rounding against the unchanged manifest oracle.

Max-pool routes reject 128-, 100- and 81-value windows instead of constructing a new
reduction tree. They return values only: no fabricated indices. The two MoE SwiGLU
cases are rejected because the available native epilogue emits MX-quantized output,
not the required floating result. Other near misses (FP8 scale formats, W4 affine
groups, RMS-softmax fusion, comparison NaN semantics, declaration-only APIs) are
explained per operator and case in the coverage file.

Compilation uses the installed AArch64 CANN 9.1.0 `bisheng`, with every output and
temporary file under `.cache/all-pairs/ops_tensor`. The produced ELF headers report
machine 183 (AArch64); `nm` confirms the bridge symbols are defined in those binaries.
No local/x86 binary was copied or linked. `build_receipts.json` contains commands,
hashes, compile logs and cache paths. The static source inventory checks 1641 vendor
source/header files against the supplied source snapshot and found no differences.
Runtime `_owned_native`/`dladdr` checks remain in the future factory path, not claimed
as performed here.

CPU-only checks can be repeated from the workspace root:

```bash
python3 -B TileOPs/benchmarks/ascend_all_pairs/ops_tensor/build.py add
python3 -B TileOPs/benchmarks/ascend_all_pairs/ops_tensor/build.py trunc
python3 -B TileOPs/benchmarks/ascend_all_pairs/ops_tensor/build.py gemm
python3 -B TileOPs/benchmarks/ascend_all_pairs/ops_tensor/build.py reduction
python3 -B TileOPs/benchmarks/ascend_all_pairs/ops_tensor/audit.py
```

The audit blocks device-capable imports, checks registration and Python syntax,
inspects original-source inclusion and ELF symbols, and evaluates metadata/layout
plans. It does not call a factory or generate tensor values. `static_checks.json` is
the CPU receipt; each `numerical_receipt` is deliberately null. `coverage.json` and
`progress.json` are also mirrored in this worker's isolated evidence directory.

Per-window and per-chunk host launch counts can be very large. No performance claim
is made. A future complete manifest validation must run under the host NPU guard,
only after the CPU-only prohibition is lifted; then verify every output, changed
inputs, input preservation, source/binary provenance and actual native launches.

`oracle_notes.json` separately records one source-only observation: the SSD coupling
reference multiplies its upper triangle by zero, which would retain NaNs if nonfinite
inputs were introduced. Current generated inputs are finite; no failure is claimed,
and the original oracle and tolerance were not changed.
