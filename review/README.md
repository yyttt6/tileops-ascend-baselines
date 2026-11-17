# Baseline performance development

Start with the [Chinese handoff](../docs/handoffs/BASELINE_PERF_REVIEW_20261117.zh.md).

```bash
python3 -m review.doctor
python3 -m review.bench catalog --op TopKMaskFwdOp
python3 -m review.bench --help
python3 -m review.compare --help
```

Edit `baselines/` or write a `build(case)` factory. Keep `frozen/` and the
unchanged public TileOPs denominator separate. Historical acceptance is bound to
the exported original source hashes, not to subsequent edits or a different host.

`bench prepare` and `bench measure` explicitly initialize NPU. The commands above
are CPU-only. No remote dispatch or internal card-management protocol is included.
`local-runs/`, inputs, raw profiles and compiled objects must remain outside Git.
