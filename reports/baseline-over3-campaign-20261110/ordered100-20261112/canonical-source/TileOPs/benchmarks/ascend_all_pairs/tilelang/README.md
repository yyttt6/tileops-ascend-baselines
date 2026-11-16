# TileLang all-pairs adapters

`PROVIDERS` in `__init__.py` is the lazy integration entry point. Importing it does
not import Torch, TileLang, or initialize an NPU. A factory accepts the original
`benchmarks.api.Case` and returns an original-library `Candidate`.

The authoritative incremental results are in
`.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang/coverage.json`.
The report includes all operators and cases in the current shared inventory,
including explicit unsupported, unmeasured, numerical-failure and deployment
states. An absent native API is a limitation of the inspected source and allowed
adaptations, not a permanent claim that the operation is impossible.

## Implementation boundaries

- `adapters.py`: original Ascend GEMM, Add and RMSNorm APIs. Batch/group slicing,
  convolution window layouts, constant-coordinate bias, and constant projection
  coefficients are transformed on every call. Runtime metadata is read on every
  call; only compiled kernels are cached.
- `attention.py`: unmodified original BF16 Ascend attention, per-head packing,
  page gather, feature padding, and decode window
  crop. Repeating every KV entry equally handles non-tile-aligned context without
  altering relative softmax mass. The underlying kernel is noncausal. `attention_crops.py` adds exact per-query
  causal/window context crops; distinct context lengths may be compilation
  limited. Softcap, RoPE and scaled FP8 variants require other original APIs.
  Earlier FP16 dtype rewriting also changed the original softmax probability
  storage precision. That route has been withdrawn as outside the permitted
  adaptations; its numerical receipts remain preserved but do not count toward
  coverage. See `attention_scope_correction.json`. FP16-to-BF16 input casting
  is not lossless and is not substituted.
- `fp8.py`: FP8 values widen exactly to BF16; per-tensor scales bind the original
  native DeepGEMM runtime `alpha`. Per-block arbitrary FP32 scales are not rounded
  to UE8M0. Original FP32 accumulation input supplies optional bias.
- `microblocks.py`: unchanged fixed-size instruction regression kernels, called
  over independent slices. There is no replacement batched device kernel and no
  claim that these many host launches are fast.
- `reduce.py`: original 2x128 native min/max reducer, repeated for partial extrema;
  boolean all/any use exact 0/1 representations. Nonboolean truth conversion is
  not supplied by a Torch arithmetic fallback.
- `sum_reduction.py`: original native sum reducer for the FP32 case whose GEMM
  accumulation order exceeded the unchanged tolerance.
- `pool.py`: fixed, adaptive and ragged sequence windows feed original GEMM or
  native max reducers. Divisors use actual window extents, including ceil-only
  padding and explicit overrides. Indices-output contracts are not replaced by
  value-only results.
- `bit_layout.py`: original XOR flips IEEE sign bits for Neg; boolean And/Or
  select original clamp extrema on exact 0/1 representations. Nonboolean truth
  tests are not computed by Torch.
- `fma.py` retains an untested initial mapping for audit only; registered Mul/Sub
  select original `merging_kernel(vmul/vsub)` directly. This avoids the signed-zero
  difference of an FMA-plus-zero multiplication shortcut.
- `merge_native.py`: complete original predicated SIMD API for native SFUs,
  integer bitwise operations and masked bit copies. Its macro and kernel bodies
  are extracted unchanged with their original operation lists.
- `truth.py`: original unsigned masks perform nonzero conversion; native bitwise
  operations handle floating sign bits, integer limbs and both complex components
  without narrowing FP64 bits. Original reducers handle
  All/Any/CountNonzero. No external arithmetic truth conversion is used.
- `leaky.py`: calls the original fixed0.1 LeakyReLU API twice for current
  slope0.01 cases, preserving its complete kernel and original N=64 constant.
- `composed.py`: original native arithmetic/SFU API compositions for hard
  activations, interpolation, runtime per-channel PReLU weights and sigmoid/SiLU.
  Native kernel bodies are unchanged; all current inputs/intermediates are
  processed inside `run`. These routes require complete case validation and do
  not inherit passes from their component operations.
- `moe_activation.py`: keeps expert GEMM accumulators in original FP32 storage,
  applies the original native SiLU composition, then casts the final result.
  The non-activation path delegates unchanged to the already tested factory.
- `diagnostics.py` is deliberately excluded from `PROVIDERS`: it records actual
  generic-kernel backend failures, without presenting a blocked kernel as an
  executable registered implementation.

Torch is used for allocation, lossless dtype conversion, layout/padding,
metadata binding and output placement. The target arithmetic remains in original
TileLang kernels. Native bodies, schedules, manifest tolerances and oracles are
not patched. Extracted modules and JIT artifacts live only in the isolated cache.

## Validation

The local runners call the unchanged shared `ascend_classic_check` in-process.
They retain complete manifest output comparison, input preservation, Add special
values/signed-zero checks, FP8 scale-floor checks and FP8 mutation-detection
self-tests. `batch_traced.py` also observes successful original
`JITKernel.__call__` launches and stores generated-kernel hashes in a sidecar.
`artifacts.py` matches those hashes to isolated-cache code and executable hashes
in `compiled_artifacts.json`.
This observation delegates launch arguments and execution unchanged.

`complex_check.py` addresses the observed NPU `IsClose` rejection of complex
inputs. It delegates to the unchanged original zero-tolerance input comparator
on exact CPU copies, checks device/dtype/shape/layout before transfer, and tests
unchanged special values plus real and imaginary mutations before enabling the
compatibility path. It leaves the original oracle and every output check intact.
The local runner records this compatibility source and its selfcheck receipt;
a shared checker needs the same input-comparison capability for complex cases.
Original failing receipts remain in `verifier_issues.json`.

`batch_state.json` records each full case in flight and its completion. A device
synchronization fault terminates the batch before further device cleanup/cases.
`reference_check.py` uses the shared checker's `cpu_reference` transport helper
to execute the unchanged adaptive-max manifest reference on exact CPU inputs.
The original NPU reference caused a CANN SIMT stack overflow, independently
attributed by driver kernel name, symbol and PC in `oracle_adaptive_max_fault.json`.
Candidate computation stays on NPU, and full output/input checks and tolerances
remain unchanged. The sidecar and Candidate adaptation identify this execution
device compatibility explicitly; no shared oracle source is edited.

The actual current queues and reservation are in `progress.json`. Queue020 is
the fresh-process AdaptiveMax CPU-reference validation plus complex checks, with
`ASCEND_LAUNCH_BLOCKING=1`, a300-second guard and240-second batch allowance.

Receipts are immutable. New attempts get a new filename. `source_snapshots/`
retains own adapter revisions; historical classic receipts stay read-only and are
reused only for the same old helper with matching recorded hashes. Generic source
constraints propagated to other cases are explicitly static, not device passes.

Run only through the host lock, for example from the workspace root:

```bash
python3 tools/npu_guard.py run --owner allpairs/tilelang --wait 3600 --timeout 900 -- \
  .tools/baselines/baseline-run tilelang python -B \
  TileOPs/benchmarks/ascend_all_pairs/tilelang/batch_traced.py \
  --queue .agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang/queue021.json \
  --seconds 800
```

The 900-second allowance is for batches containing original native compilation.
The foreground batch stops between cases; a per-case wall allowance records a
failure rather than crediting partially computed outputs. Exit codes 75/76/78
from the host guard must not be bypassed. No performance result is claimed.

`finish_queue.py` is a CPU-only foreground coordinator for repeated bounded
batches. Run it outside a guard: it puts each complete device subprocess through
its own `npu-guard`, waits for completion, refreshes the evidence indexes, and
stops on any nonzero guard exit or lack of progress. It never retries an existing
failed receipt. For a queue that still requires original kernel compilation:

```bash
python3 -B TileOPs/benchmarks/ascend_all_pairs/tilelang/finish_queue.py \
  --queue .agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang/queue021.json \
  --timeout 900 --compilation-required
```

Regenerate the CPU-only source inventory and coverage report with:

```bash
python3 -B TileOPs/benchmarks/ascend_all_pairs/tilelang/report.py
```
