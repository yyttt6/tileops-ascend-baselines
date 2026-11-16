# DeepSelect all-pairs adapters

This package exports `PROVIDERS[exact_op_name](case) -> Candidate`. Package import
uses only the standard library. Runtime dependencies and the owned native build
are resolved inside the factory. No device execution occurred in this task.

The source audit covers all **205 operators / 2062 current inventory cases**.
The assignment initially listed 1982 cases; the shared inventory was expanded
during the task with 80 missing-factory manifest rows. Both the inventory hash
and a snapshot are retained. The original manifests and oracles are unchanged.

| Provider | Semantically supported cases | Adaptation |
| --- | ---: | --- |
| AmaxFwdOp | 4 | Fold reduction axes; native TopK `k=1` value |
| AnyFwdOp | 4 | Bool-only exact 0/1 widening, maximum, bool representation |
| MaximumFwdOp | 10 | Broadcast and pack each operand pair as a row |
| LogicalOrFwdOp | 4 | Bool-only operand pairs, exact 0/1 widening |
| ReluFwdOp | 3 | Pack each input value with a constant zero |
| MaxPool1dFwdOp | 3 | Pad and pack strided/dilated windows |
| MaxPool2dFwdOp | 6 | Pad and pack strided windows, including ceil mode |
| MaxPool3dFwdOp | 3 | Pad and pack 3D windows |
| AdaptiveMaxPool2dFwdOp | 4 | Exact floor/ceil window boundaries, including nondivisible shapes |

These **41 cases have semantic_status `supported_unverified` and current status
`deployment_blocked`**. The remaining 2021 cases have individual `unsupported`
reasons tied to inspected source and manifest contracts. No case is `verified`
or `numerical_failure`: there are no device/numerical receipts for this run.
Unsupported is scoped to this source snapshot and the allowed adaptations.

Every computed maximum comes from the original
`deep_select.topk(..., topk=1, return_value=True)[0]`. Torch is used only for
allocation, copies, shape/stride/broadcast changes, padding and lossless dtype
representation. Input-dependent work runs on every invocation, inside `run`.
No input data or transformed tensors are cached by the factory. Row batching
limits temporary packing storage and does not change the per-row selection.

FP16 uses the established bit-preserving view into the BF16 selector: both
16-bit formats have the same finite sign-magnitude ordering. Arbitrary FP32
cannot use this route losslessly. NaN propagation is outside the finite manifest
input domain because the original kernel aborts on BF16 NaNs. Original `+0 > -0`
selection does not promise bit-identical returned zero signs. Boolean mappings
operate solely on exactly representable 0 and 1.

Index-output operators are not registered. Long-row page shuffling changes
which equal maxima are selected. Short rows scan sequentially, but native
`+0 > -0` still disagrees with the required first index of numerically equal
zeros. Pooling indices cannot be dropped from the full result. TopKSelect's
FP32 inputs and sampling masks' missing full-output mathematics are documented
separately; neither is replaced by torch computation or a CUDA implementation.

## Deployment evidence

Only compiler processes were executed. The unmodified Ascend kernel failed on
CANN 9.1.0 because these newer public APIs are undeclared:

- `asc_pack_to_low`
- `asc_duplicate_highest`
- `asc_squeeze_and_storeunalign_init`
- `asc_squeeze_and_storeunalign`
- `asc_squeeze_and_storeunalign_finalize`

The original host binding also failed on `std::format`. `build_cpu.py` makes a
private host-source copy replacing only that exception-message expression with
equivalent variadic `TORCH_CHECK` arguments. This copy compiled successfully.
The original vendor files remain unchanged. No speculative vector-instruction
shim, altered kernel schedule, foreign binary, or runtime fallback was added.

`build_cpu.py` can reproduce a full native build in the task's `.cache` directory
when compatible headers/toolchain are available. It reads the libc10 ABI from
ELF symbols and queries Python sysconfig under `-I -S`; it never imports torch,
torch_npu, DeepSelect, or ACL. The current kernel failure prevents linking.

The runtime loader expects `.cache/all-pairs/deepselect/native/build_receipt.json`
(or a workspace-local `DEEPSELECT_ALL_PAIRS_BUILD`). It checks vendor and
compatibility source hashes, original Python wrapper hashes, actual extension
hash, host/ELF architecture, and the loaded native module path. An installed
namesake is rejected. The receipt is a build record, not device validation.

## Evidence and static audit

- `coverage.json`: every operator/case, source file/line/hash references, exact
  reasons, registration/adapter path, factory availability and null numerical,
  input-protection and runtime-ownership receipt paths.
- `static_checks.json`: import isolation, AST/syntax checks, 41 output/layout
  plans, complete coverage, unchanged manifest hashes, and permitted torch calls.
- `source_catalog.json`: owned API/source fingerprints; test references, CUDA
  and embedded dependencies are explicitly excluded from Ascend attribution.
- `deployment.json`: original kernel/API failures and the successful host-only
  diagnostic compatibility compilation.
- `progress.json`: completion checkpoint and issues for the root supervisor.

The audit is reproducible without any runtime import:

```bash
python3 -I -S TileOPs/benchmarks/ascend_all_pairs/deepselect/audit.py
```

Copies of the reports, compiler logs, inventory snapshot and append-only audit
history are in the task-owned `ALL_PAIRS/deepselect` evidence directory. Missing
workload factories are reported independently from mathematical support. Their
current AST list is retained alongside the shared inventory's earlier flags,
because other workers were adding factories during this audit. None of the
historical x86/CANN9.2 verification results is claimed as ARM64 evidence.

Future authorized device validation must check the complete original manifest
outputs, input preservation and actual DeepSelect native-call ownership. Such
validation and all benchmarking were forbidden for this task.

## Local x86 integration follow-up

The root integration built the original kernel and the diagnostic-only host copy
on x86_64 with CANN 9.2.0 beta.2. The build is `compiled_unverified`; no new device
acceptance is implied. Its receipt is
`.cache/all-pairs/deepselect/native/build_receipt.json`, with the full command and
logs in `.cache/all-pairs/deepselect/builds/20261021T113611889996Z/report.json`.

The C++20 device frontend rejected constrained `std::pair` declarations in GCC
12/13 headers. GCC 11 development packages were downloaded and extracted only
under this task's cache; no system compiler or vendor source was replaced. Package
hashes are in `.cache/all-pairs/deepselect/gcc11-package-provenance.json`.
`--gcc-toolchain .cache/all-pairs/deepselect/gcc11-sysroot/usr` selects these headers.
The kernel still uses C++20 and its original schedule. `build_cpu.py` now selects
the CANN architecture directory from the host and accepts this optional toolchain.
The earlier ARM64/CANN 9.1 findings above remain separate evidence.
