#!/usr/bin/env python3
"""CPU-only receipt/source/native artifact consistency audit."""
import collections
import ast
import hashlib
import json
import pathlib
from datetime import datetime, timezone

WS = pathlib.Path(__file__).resolve().parents[4]
EVIDENCE = WS / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilekernels'
CACHE = WS / '.cache/all-pairs/tilekernels/tilelang'
ALGORITHM_MODULES = {'adapters.py', 'extended.py', 'pointwise.py', 'reductions.py', 'dequant.py', 'moe.py', 'selection.py', 'additional.py', 'pooling.py', 'select_extra.py', 'reciprocal.py', 'activation_composed.py', 'pooling_layout.py', 'pooling_max.py'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return str(p.relative_to(WS))
def read(p): return json.loads(p.read_text())
artifacts = collections.defaultdict(list)
for source in CACHE.rglob('device_kernel.asc'):
    binary = source.with_name('executable.so')
    if binary.exists():
        artifacts[sha(source)].append({'source': rel(source), 'source_sha256': sha(source),
                                      'binary': rel(binary), 'binary_sha256': sha(binary),
                                      'cache_manifest': rel(source.with_name('manifest.json'))})
def launch_artifacts(k):
    binary = pathlib.Path(k.get('binary') or '')
    if k.get('binary_sha256') and binary.is_file() and sha(binary) == k['binary_sha256']:
        source = binary.with_name('device_kernel.asc')
        if source.is_file() and binary.is_relative_to(CACHE):
            return [{'source':rel(source), 'source_sha256':sha(source),
                     'binary':rel(binary), 'binary_sha256':sha(binary),
                     'cache_manifest':rel(binary.with_name('manifest.json')),
                     'match_method':'actual cached-library path and runtime-captured binary hash'}]
    return artifacts[k['generated_source_sha256']]
coverage = read(EVIDENCE/'coverage.json')
active = {r['receipt'] for r in coverage['cases'] if r['status']=='verified' and r.get('verification_origin')=='new_device_receipt'}
def relevant_modules(receipt):
    pending = [WS/f for f in receipt.get('source_sha256',{}) if pathlib.Path(f).name in ALGORITHM_MODULES]
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file(): continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependency = path.parent / (node.module.split('.')[0]+'.py')
                if dependency.name in ALGORITHM_MODULES: pending.append(dependency)
    return {rel(p) for p in found}
rows = []
archived = []
for rp in sorted((EVIDENCE/'receipts').glob('*.json')):
    receipt = read(rp)
    if rel(rp) not in active:
        archived.append({'receipt':rel(rp),'status':receipt.get('status'),'reason':'Not the currently accepted receipt for this case; raw evidence preserved.'})
        continue
    verification = receipt.get('verification') or {}
    changed = [f for f,h in receipt.get('source_sha256',{}).items()
               if not (WS/f).exists() or sha(WS/f) != h]
    dependencies = relevant_modules(receipt)
    changed_algorithms = [f for f,h in receipt.get('provider_sha256',{}).items()
                          if f in dependencies and (not (WS/f).exists() or sha(WS/f) != h)]
    reference_problems = []
    if receipt.get('reference_placement_adaptation'):
        if sha(WS/'TileOPs/workloads/pool.py') != receipt['reference_source_sha256']:
            reference_problems.append('original reference source changed')
        diagnostic = WS/receipt['oracle_failure_diagnostic']
        if sha(diagnostic) != receipt['oracle_failure_diagnostic_sha256']:
            reference_problems.append('oracle diagnostic fingerprint changed')
        proof = read(diagnostic)
        if proof['candidate_native_calls'] != 0 or proof['status'] != 'original_reference_device_failure' or not proof['cpu_reference_same_function_passed']:
            reference_problems.append('missing independent original-reference failure proof')
    kernels = [{'kernel_name': k['name'], 'launch_calls': k['calls'], 'target': k['target'],
                'vendor_stack': k['vendor_stack'],
                'matching_compiled_artifacts': launch_artifacts(k)}
               for k in receipt.get('native_kernel_launches',[])]
    rows.append({'receipt': rel(rp), 'op':receipt['op'], 'case_id':receipt['case_id'],
                 'complete_outputs': verification.get('checked_outputs') == verification.get('total_outputs') and verification.get('total_outputs',0)>0,
                 'input_preservation_requested': receipt.get('input_preservation_requested'),
                 'changed_candidate_sources': changed, 'changed_algorithm_modules': changed_algorithms,
                 'reference_device':receipt.get('reference_device'), 'reference_problems':reference_problems,
                 'kernels': kernels})
problems = []
for r in rows:
    if not r['complete_outputs'] or not r['input_preservation_requested']: problems.append([r['receipt'],'incomplete verification'])
    if r['changed_candidate_sources'] or r['changed_algorithm_modules']: problems.append([r['receipt'],'changed algorithm source'])
    if r['reference_problems']: problems.append([r['receipt'],r['reference_problems']])
    if not r['kernels']: problems.append([r['receipt'],'missing native launch'])
    for k in r['kernels']:
        if not k['vendor_stack'] or 'ascend' not in k['target'].lower() or len(k['matching_compiled_artifacts']) != 1:
            problems.append([r['receipt'],k['kernel_name'],'ambiguous/missing native artifact attribution'])
audit = {'schema':'tilekernels-native-artifact-audit/v2', 'at':datetime.now(timezone.utc).isoformat(),
         'method':'Match runtime-captured loaded-library path/hash directly, or cold-compile generated source SHA256, to isolated cache executable. Older trace records aggregate calls by name/source hash; cached-source-empty records prove the captured native library, not every distinct signature in that aggregate. Audit only current accepted coverage receipts, preserve superseded attempts separately; revalidate Candidate source hashes and recorded transitive local algorithm imports; registry/report/harness metadata changes are separate from implementation changes.',
         'summary':{'verified_receipts':len(rows),'native_source_variants':len({a['source_sha256'] for r in rows for k in r['kernels'] for a in k['matching_compiled_artifacts']}),'problems':len(problems)},
         'problems':problems,'receipts':rows,'archived_attempts':archived}
(EVIDENCE/'native_artifact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit['summary']))
if problems: raise SystemExit(1)
