# ops-math all-pairs adapter

`PROVIDERS` is a lazy op-name → Case factory mapping containing accepted executable routes. `CANDIDATES` holds private trial factories; `enabled.json` is updated after native acceptance. Importing the package uses only the standard library. All device imports, loading and invocation must run under `npu-guard --owner allpairs/ops_math`.

The implementation uses unchanged local `ascend_baselines/ops-math` kernels through the original graph compiler API, existing native ACLNN Add, and the complete LayerNorm fast-launch example. Layout/scalar/axis packing occurs on every invocation. No tensor arithmetic is implemented as a Torch fallback. Original workload reference, numerical policy, output count, input protection, and Add edge checks are reused from `scripts/ascend_classic_check.py`.

Builds are isolated in `.cache/all-pairs/ops_math`. The selected vendor is `install/vendors/allpairs_math_math`; source its `bin/set_env.bash` after `baseline-run ops-math` and before runtime imports. LayerNorm has a separate original-source binary and build receipt in `layernorm/`.

Evidence, progressive statuses, exact manifest jobs, logs and numerical receipts live in `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_math`. `coverage.json` defines its schema and distinguishes numerical pass without ownership, accepted verification, deployment blockage, numerical failure, semantic mismatch and dependency-only APIs. `accept.py` matches own-vendor graph compilation to actual kernel launches and archives the JIT binary. Historical Add evidence is reused only after revalidating every source hash, original full-output/edge receipt and root ownership review.

`run_cases.py` invokes the common checker within one reservation; `queue.py` is a CPU-only foreground supervisor which acquires a separate bounded reservation for each batch. Do not run it nested inside another reservation. Neither script modifies the shared provider registry, vendor source or historical results.

Additional original-API routes include window ReduceMean/Max or ArgMaxWithValue pooling, weighted ReduceSum for dot products and MHCPost, Mul for symmetric INT8 dequantization, Cdist against zero for supported vector norms/counts, and TopKV2 over cropped selection windows. Native input data, reduction weights and divisor tensors are rebuilt per call. These are private candidates until independently accepted; the coverage file, not source availability, records tested status.
