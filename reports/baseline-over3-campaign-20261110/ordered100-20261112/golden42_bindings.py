"""Explicit untimed golden selection, after the original baseline is constructed."""
from pathlib import Path
import hashlib
import json

from golden42_reference import SCOPE, corrected_reference_for_case


def bind(ordinal, meta, canonical, torch, args, sources, evidence_dir):
    assert ordinal in SCOPE
    root = Path(__file__).resolve().parent
    manifest = root/'GOLDEN42_CERTIFICATES.json'
    certificates = json.loads(manifest.read_text())
    assert certificates['status'] == 'all42_certified'
    record = next(r for r in certificates['rows'] if r['ordinal'] == ordinal)
    reference = corrected_reference_for_case(ordinal, meta, manifest)
    # New mathematical oracle only. canonical moves CPU results back to the
    # original output device without changing dtype; baseline impl is untouched.
    wrapped = canonical.cpu_reference(torch, reference, args)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for p in (Path(__file__), root/'golden42_reference.py', manifest):
        sources[str(p)] = sha(p)
    evidence = {
        'ordinal':ordinal, 'key':meta['key'],
        'original_reference_policy':meta['reference_policy'],
        'selected_reference_policy':'audited_cpu_fp64_exact42_v1',
        'payload_sha256':meta['payload_sha256'],
        'descriptor_sha256':meta['descriptor_sha256'],
        'certificate':record, 'manifest_sha256':sha(manifest),
        'old_prepare_and_failures_unchanged':True,
        'baseline_callable_kernel_precision_checker_tolerance_unchanged':True,
    }
    target=Path(evidence_dir)/'GOLDEN_CORRECTION.json'
    assert not target.exists()
    target.write_text(json.dumps(evidence,indent=2)+'\n')
    return wrapped
