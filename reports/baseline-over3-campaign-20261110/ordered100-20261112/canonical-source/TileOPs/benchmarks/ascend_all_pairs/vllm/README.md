# vllm-ascend all-pairs adapters — CPU-only delivery

Entry: `benchmarks.ascend_all_pairs.vllm.PROVIDERS`. Each factory accepts the original `Case` and returns `benchmarks.ascend_baselines.Candidate`. Importing the package is stdlib-only; tensor runtimes, original vendor modules and compilation are deferred to factories.

This delivery reviews all **205 operators / 2062 inventory cases**, including the 80 cases added to the shared inventory during work. It registers **61 concrete operator factories**. **815 cases have implemented semantic routes**, and **1247 have an explicit owned-API/contract exclusion**. No case is numerically verified. The supplied ARM64 Python3.12 runtime has no Triton package, and the native extension/custom OPP is not deployed; consequently all 815 implemented cases currently have `status=deployment_blocked` and `semantic_status=supported_unverified`.

`coverage.json` is the per-operator/per-case result. It includes source file/line/SHA256 evidence, adapter location, original API, exact exclusion/block reason, workload-factory availability and empty numerical receipts. `source_catalog.json` identifies the inspected owned source surface. `inventory_snapshot.json` freezes the input to the receipt. `static_checks.json` is strictly a CPU static receipt.

The adapters cover arithmetic, gates, reductions, norms, average pooling, GEMM/BMM/grouped GEMM, pointwise convolutions, expert MLPs, RoPE variants, KDA/GDN/DeltaNet, routing, native dense/packed/paged attention, MLA/NSA, complete attention-state merging, full cache gather/write results, paged append and causal convolution. Cases outside a concrete route are rejected by metadata guards. No Torch fallback computes a requested numerical operator.

Original vendor files and kernel schedules are unchanged. `runtime.py` rebinds framework imports into isolated modules under `.cache/all-pairs/vllm/runtime`. The DCP merge path selects an intact original JIT definition to avoid distributed registration; static AST checks compare the entire definition. Candidate hashes include original sources, deployment wrappers and generated import-only modules. All input-dependent transforms, metadata reads and mutable-buffer clones occur inside `run`.

RoPE frequency-table helpers construct API state only; the original vendor kernel performs rotation. Sum uses native mean followed by native scalar multiplication, and pooling groups complete windows for native reduction. These compositions may differ in floating-point rounding and must pass the original unchanged oracle. Cumsum's original full vector-chunk route may exceed compiler/device resources at very long widths. Repeated per-request/query dispatch can be expensive; all launches and data movement are included in the candidate call.

Run the permitted audit without importing device libraries:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/vllm/audit.py
```

The audit compiles Python syntax in memory, imports registration behind a hard device-runtime import blocker, checks the exact 205-op partition and native Python API arguments, compares import-rewrite ASTs, and checks every case's evidence/status. It refreshes this package's inventory snapshot/coverage from the shared inventory. It does not invoke factories, build kernels, create device tensors or benchmark.

Deployment remains external to this CPU-only task. Native factories expect an independently built ARM64 extension at `.cache/all-pairs/vllm/install/vllm_ascend/vllm_ascend_C.abi3.so` and custom OPP under `.cache/all-pairs/vllm/install/vendors/custom_transformer`. Optional `VLLM_ALL_PAIRS_EXTENSION` / `VLLM_ALL_PAIRS_OPP` overrides must resolve inside this library's source/build directories. `ASCEND_CUSTOM_OPP_PATH` must be set before process startup. The loader checks ELF architecture and does not copy/link another machine's binary. The gating source snapshot additionally lacks `csrc/moe/moe_gating_top_k/op_kernel/arch35/moe_gating_top_k_regbase.h`. CausalConv1dV310 is FP16/ascend310p-only, so legal registration on the current target is unresolved. Complete-LSE attention also needs the original `sparse_flash_mla_metadata` AICPU component; its attention OpDef accepts BF16 only.

The frozen shared inventory still reports missing factories for `DSAPagedFwdOp`, `GemmW4A8FwdOp`, `HadamardTransformFwdOp`, `MLAKVCacheWriteFwdOp`, `MLAPagedFwdOp`, and `MLAVarlenFwdOp`. This package implements legal MLA cache-write/varlen subsets, but does not fabricate their workload execution. Missing-factory status is separate from the mathematical/API decision. The shared manifest, oracle, tolerances and registry have not been edited.

No NPU initialization, import, compilation, test, benchmark or profiling was performed. Historical x86/CANN9.2 verified results are not reused as this host's evidence. Future numerical verification must check all outputs, original input protection and real owned-kernel launch attribution under the host's authorized device protocol.
