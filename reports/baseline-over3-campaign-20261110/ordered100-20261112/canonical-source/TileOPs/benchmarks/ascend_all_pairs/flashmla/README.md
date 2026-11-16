# FlashMLA: 2026-10-22 continuation

Current evidence: 4 original cases on the current relevant source, 69 passes bound to preserved earlier source versions. New receipts await root review. Coverage spans all 205 operators and 2062 original cases; unsupported rows are not passes. See `resume22_validation.json`, `coverage.json`, and `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/resume22_native/REPORT.md`.

The predecessor actually completed all four bool LogicalOr cases with complete guard0/native proof. This worker rechecked and sealed those records. The older 69 cases include nine original Flash reruns already accepted by root on their preserved earlier package. No new Flash NPU execution was performed here.

Original vendor algorithms, shapes, seeds, factories, oracle and tolerances are unchanged. W4A8 original 12-case oracle holds remain. No performance claim is made. The older narrative below is retained as historical context.

---

# FlashMLA current continuation status — 2026-10-21

The current package has **4 freshly verified original cases** and **69 prior-version passes**. Prior passes retain their exact evidence and are not inherited by changed Python sources. New receipts await root review. `continuation_validation.json`, `progress.json` and `coverage.json` contain the current counts across all 205 operators / 2062 cases for this library.

This update adds bool-only LogicalOr through exact 0/1 BF16 layout and the existing original sparse-prefill maximum statistic. Nine formerly held original cases were fully revalidated with guard0 on the preserved prior package before this source change. New OR results are individually recorded in current coverage.

The vendor kernels, original oracle, tolerance, factory, RNG seed, shapes and precision are unchanged. No performance measurement is claimed. W4A8 original 12-case oracle gaps remain held. The old CPU/build narrative below is preserved as historical context; its old zero-pass/deployment statements do not describe current status. Detailed evidence: `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/local_native_continuation/REPORT.md`.

---

# FlashMLA all-pairs CPU integration

This isolated package exports `PROVIDERS[exact_op_name](case) -> Candidate`.
Its import path uses only the standard library. Factories reject unsupported
contracts before loading the native extension. Every accepted implementation
calls FlashMLA's original `sparse_prefill_fwd`; there is no torch numerical fallback.

The source review covers all **205 operators and 2062 current inventory cases**.
The parent expanded the shared inventory during this task; all **1982 original
cases** remain covered. An immutable inventory snapshot and hash are recorded
in `coverage.json`. No missing-factory cases were invented by this worker.

There are **22 providers and 69 semantically supported, unverified cases**.
Their current operational status is **deployment_blocked**. The other **1993
cases** lack a legal path in this source snapshot under the stated restrictions.
All device-run, numerical-pass and numerical-failure counts are **zero**.
Historic x86 results are not ARM64 test results.

The adapters include:

- Dense, varlen, paged and append-to-paged GQA, with windows, grouped heads and
  native sinks where the original contracts permit them; DSA, NSA selected-block
  and compressed attention; MLA varlen with already-encoded shared RoPE keys.
- Exposed native max/LSE, uniform native weighted means, boolean Any through the
  exact bool-to-BF16 0/1 representation, spatial/adaptive maximum pooling,
  supported average pooling and sequence chunk means.
- Maximum and ReLU through the same basis-query maximum route as Amax, using
  broadcast/stack or a zero padding value.
- BF16 BMM with contraction width at most512: each B column is the only selected
  key, so native `max_logits` is that complete QK dot product. Larger K is rejected;
  the adapter never splits the contraction or adds partial sums.

All packing, metadata reads, cache clones, padding, copies and output crops occur
inside `run`. Reduction rows share native query heads/columns to reduce workspace;
the vendor kernel schedule is unchanged. Host-reading adapters are marked
`host_roundtrip`. Pooling and other padded routes may be expensive; no latency
measurement or performance claim has been made.

Runtime guards check every supplied manifest input and returned tensor's shape
and dtype. Append-to-paged GQA returns the full cloned K/V caches along with O;
NSA compressed and MLA varlen return O and LSE. These checks prevent a shortened
workload or partial result from being accepted. Actual numerical comparison and
input-preservation checks still require a separately authorized device run.

`deployment_review.json` records two CPU-only compilation attempts. The isolated
build stages vendor sources under `.cache/all-pairs/flashmla`, and changes only
the host diagnostic formatter's `constexpr std::string` declaration to `inline`
for GCC11 compatibility. All kernel sources and the original vendor tree are
unchanged. Six host translation units compile. The first device translation
unit fails because CANN9.1 lacks seven required identifiers at14 error locations:
quarter-selection conversion constants, broadcast/unpack load variants and typed
L2 cache modes. No unproven intrinsic substitutions were added, no DSO was copied
from another host, and no generated binary was loaded.

The loader requires a successful local build receipt, matching source/staged
hashes and matching AArch64 ELF bytes before importing torch/torch_npu and the
extension. Current factories therefore stop with a concrete deployment error.
`build.py` never imports those runtimes and never probes or initializes a device.

`oracle_issues.json` independently identifies two current dense-GQA chunked-prefill
cases whose factory selects `GQADenseDecodeCall`, generating S_q=1 despite the
multi-query manifest. The original factory and oracle were left unchanged.
`coverage.json` also records the current shared factory gaps, including MLA varlen.

The following commands perform CPU-only review; they do not run numerical tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python TileOPs/benchmarks/ascend_all_pairs/flashmla/audit.py
PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python TileOPs/benchmarks/ascend_all_pairs/flashmla/check.py
```

`check.py` blocks runtime imports, checks syntax and lazy registration, verifies
factory decisions against the complete inventory, tests integer pooling boundary
plans and manifest guards using inert metadata records, and checks unchanged
manifest/vendor/kernel hashes. It never labels these checks as numerical passes.

Files for collection: the Python modules, this README, `coverage.json`,
`progress.json`, `static_checks.json`, `deployment_review.json`, and
`oracle_issues.json`. Detailed build commands and diagnostics remain under the
library's isolated cache. JSON reports are also mirrored to the authorized
FlashMLA evidence directory; no shared registry or other task memory was edited.
