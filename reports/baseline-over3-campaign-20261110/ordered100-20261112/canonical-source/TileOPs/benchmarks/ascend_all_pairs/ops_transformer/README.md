# Isolated ops-transformer expansion

`PROVIDERS` maps exact original op names to lazy `factory(case) -> Candidate` functions. Importing the provider module does not import Torch or initialize a device. All native computations use the original ops-transformer source; runtime packing, metadata reads, casts and copies are inside `run`.

From `/root/dyq/workspace`, run a new immutable numerical receipt:

```bash
npu-guard run --owner allpairs/ops-transformer-continuation --wait 7200 --timeout 300 -- \
  bash TileOPs/benchmarks/ascend_all_pairs/ops_transformer/run.sh \
  --op BmmFwdOp --case-id square-128-float16 --library ops-transformer \
  --output /root/dyq/workspace/.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer/manual-new.json
```

Use hard300 and a complete-case soft60 boundary. A genuine JIT case requiring900 needs prior actual logs; prefer CPU-only precompilation outside the guard. Never submit more than one live/pending reservation for the owner. `check.py` reuses the original complete-manifest checker, its input-protection contract and its existing special Add/FP8/RoPE/Delta controls. Local RoPE variants additionally receive an independent CPU rotation check with the exact original device-generated coefficients. A numerical pass is not an ownership acceptance: `coverage.json` requires a separate accepted `.ownership.json` receipt before marking a new case `verified`.

`build.py`, `build_causal.py`, `build_kda.py`, `build_attention.py`, and `build_mla.py` copy the unmodified original operators into independent cache directories and build the original host/JIT packages. `build_experimental.py` reuses the original-source compile/link recipe recorded in `state_passing_build.json`; source `env.sh` before invoking it. All build/output/cache/log files remain in the worker's allowed directories. `source_copy_audit.json` records source equality.

CANN log capture is retained per case as `.native.jsonl`. `ownership.py` requires a successful compile from an original transformer vendor, a matching launch by this receipt’s runtime PID within its execution interval, and matching recorded source hashes. Direct original C++ dispatch additionally checks its source registration and isolated shared object. Executed adapter snapshots preserve the exact bytes behind receipts.

`native.py` uses the original graph API. `graph_layout.py` supplies the original
GroupedMatmul typed empty-group descriptors and packed INT4 view. For FP32, `list_tensor_arg.py` supplies the exact SDK ListTensorDesc wire format because actual c09 logs proved that the graph entry passed ordinary buffers unchanged. Quantized branches remain held until separately repaired. `infer_shim.cpp` supplies
only the missing MoeFusedTopk shape/dtype registration. Both are pending device
acceptance until their individual receipts pass. The retained `aclnn_bridge.py`
was diagnostic: frontend calls selecting system built-in kernels are rejected by
the ownership audit and are no longer selected by providers.

`deployment_fix.py` removes only `-Werror` from isolated generated wrappers to
accommodate CANN 9.2 beta header warnings; before/after hashes and original wrapper
snapshots are retained. Run it only while no checker process is loading this
package. INFO logs retain compile and launch evidence; incremental capture
positions avoid repeatedly reading old native logs between cases.

The incomplete DeltaNetChunk route is unregistered: its original manifest
requires all six outputs, while ChunkGatedDeltaRule provides only output and
final state. The NSA CPU reference transport copies the bound workload's hidden
`g_slc` tensor to CPU together with its arguments; original oracle code and values
are preserved.

CPU-only incremental reporting:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/ops_transformer/ownership.py
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/ops_transformer/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/ops_transformer/make_queue.py
```

The report enumerates all 205 original manifest operators and 2062 cases. Historical verified cases are labeled separately. Implemented but untested, numerical failure, runtime/deployment blockers, unresolved mappings, dtype/kernel limits and explicit contract mismatches are separate states. Consult exact case status and receipt before use; provider registration alone is not validation. No shared registry, matrix, vendor source, original manifest or oracle is modified by this worker.

`recover_log_tail.py PID` preserves logs buffered until checker exit. Ownership
checks match the exact PID and case execution interval across capture files,
including the preserved tail; transport file boundaries do not define execution
time. `reference_deps.py` resolves the NSA CPU oracle's original `tilelang.math`
dependency directly from its unchanged installed source, avoiding the unrelated
CANN/TileLang TVM namespace collision. It is used only during reference calls.

`pointwise_extrema.py` and `prefix_sum.py` extend the accepted raw-maximum and
GroupedMatmulAdd routes to more original manifests. These adapters synchronize
and release descriptors inside `run` to bound memory for large tensors. Their
individual acceptance status remains in `coverage.json`; registration is not a
numerical claim. `build_infer_registry.py` reproducibly builds the missing graph
metadata registration, without changing numerical kernels or schedules.

`pool_indices.py` maps complete max-pool values and original spatial indices
through the original top-1 API. Both outputs retain the exact original oracle.
InstanceNorm running-statistics cases are excluded because their manifest has
three outputs and the selected native API exposes only normalized tensors.

`build_selection.py` prepares original `MoeGatingTopKSoftmaxV2` and `MoeGatingTopK` host/JIT packages. `selection_install.json` records source equality and generated-wrapper warning compatibility. V2 `renorm=1` selects raw scores for index/extrema adapters; `gating_normalization.py` binds full original sigmoid/softmax outputs and grouped expert routing. These new routes require their own full-manifest device receipts.

`routing_arithmetic.py`, `composed_activations.py`, `boolean_reduce.py` and
`fused_silu.py` use original exported routing/selection outputs. New paths keep
FP32 intermediates until the manifest output cast. Add's original x1 accumulator
path preserves signed-zero semantics; the original strict Add edge checks remain
mandatory. These paths remain unverified until their individual receipts pass.

After recovering runtime tails and refreshing ownership/coverage,
`make_pending_queue.py NEW_LABEL` creates a new queue snapshot containing only
registered `supported_untested` cases. It excludes known failures and prior
verified receipts, and visits a small case of each operation first. The earlier
explicit repair queues remain separate because those revisions have concrete
source fixes. Conditional queue items use `requires_verified_output` to gate
GroupedMatmul-dependent probes; a numerical pass alone never grants ownership
acceptance.

Continuation evidence is kept separately in `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer_continuation`. c11 proved that the diagnostic owned ACLNN frontend selected a CANN built-in GroupedMatmul binary; its guard78 result is rejected and this frontend is no longer selected. c13 returns to the original source graph entry. The Varlen RoPE, prefill RoPE/cache, and grouped GEMM/SiLU compositions now have explicit metadata dispatch, but require fresh complete original-case receipts. Existing receipts remain tied to their captured source versions.
