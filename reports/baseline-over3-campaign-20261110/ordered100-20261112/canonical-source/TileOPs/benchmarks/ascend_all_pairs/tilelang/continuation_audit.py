"""Read prior evidence without importing device libraries or promoting claims."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang'
OUT = PRIOR.with_name('tilelang_continuation')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    coverage = json.loads((PRIOR / 'coverage.json').read_text())
    frozen = {}
    def read(path):
        path = ROOT / path
        frozen[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())
    ast_checks = []
    for record in read(PRIOR / 'kernel_ast_audit.json')['records']:
        extracted = ROOT / record['extracted_file']
        funcs = [n for n in ast.walk(ast.parse(extracted.read_text()))
                 if isinstance(n, ast.FunctionDef) and n.name == record['function']]
        dumps = [ast.dump(n, include_attributes=False) for n in funcs]
        matches = []
        for source in record['original_matches']:
            original = ROOT / source['file']
            node = next(n for n in ast.walk(ast.parse(original.read_text()))
                        if isinstance(n, ast.FunctionDef) and n.lineno == source['line'])
            dump = ast.dump(node, include_attributes=False)
            matches.append({'file': source['file'], 'line': source['line'],
                            'vendor_hash_matches': digest(original) == source['source_sha256'],
                            'exact_ast_matches': dump in dumps,
                            'ast_hash_matches': hashlib.sha256(dump.encode()).hexdigest() == record['ast_sha256']})
        ast_checks.append({'extracted_file': record['extracted_file'],
                           'function': record['function'], 'matches': matches})
    correction = read(PRIOR / 'attention_scope_correction.json')
    prior_by_key = {(c['op'], c['case_id']): c for c in coverage['cases']}
    withdrawals = []
    for case in correction['affected_cases']:
        receipt = read(case['receipt'])
        current = prior_by_key[case['op'], case['case_id']]
        withdrawals.append({'op': case['op'], 'case_id': case['case_id'],
                            'receipt_status': receipt['status'], 'coverage_status': current['status'],
                            'withdrawal_effective': current['status'] != 'verified'})
    source = ROOT / correction['source']['file']
    retained = ROOT / correction['retained_bf16_source']['file']
    artifact_checks = []
    for record in read(PRIOR / 'compiled_artifacts.json')['records']:
        matching = []
        for a in record['matching_artifacts']:
            matching.append({'generated_source': a['generated_source'],
                             'source_hash_matches': digest(ROOT / a['generated_source']) == record['kernel_sha256'],
                             'executable_hash_matches': digest(ROOT / a['executable']) == a['executable_sha256'],
                             'manifest_hash_matches': digest(ROOT / a['cache_manifest']) == a['cache_manifest_sha256']})
        artifact_checks.append({'attribution': record['attribution'], 'matching_artifacts': matching})
    receipts = []
    for case in coverage['cases']:
        if case['status'] != 'verified':
            continue
        path = case.get('device_receipt') or case.get('historical_classic', {}).get('receipt')
        receipt = read(path)
        verification = receipt.get('verification', {})
        sources = []
        for name, expected in receipt.get('source_sha256', {}).items():
            f = ROOT / name
            current = digest(f) if f.is_file() else None
            snapshot = PRIOR / 'source_snapshots' / (expected + '.py')
            sources.append({'file': name, 'recorded_sha256': expected, 'current_sha256': current,
                            'current_matches': current == expected,
                            'own_snapshot_matches': snapshot.is_file() and digest(snapshot) == expected})
        attribution = read(case['native_attribution']) if case.get('native_attribution') else None
        receipts.append({'op': case['op'], 'case_id': case['case_id'], 'receipt': path,
                         'status': receipt.get('status'), 'verification': verification,
                         'full_output_count_matches': bool(verification.get('checked_outputs')) and
                            verification.get('checked_outputs') == verification.get('total_outputs'),
                         'source_bindings': sources,
                         'native_evidence': 'successful_original_JIT_host_calls' if attribution else 'prior_factory_source_only',
                         'native_successful_calls': attribution.get('total_successful_calls') if attribution else None,
                         'driver_load_launch_directly_observed': False})
    delivery = read(PRIOR / 'delivery.json')
    delivered_sources = [{**r, 'matches': digest(ROOT / r['path']) == r['sha256']} for r in delivery['source_files']]
    reservation = read(PRIOR / 'reservation020_result.json')
    reservation['log_hash_matches'] = digest(ROOT / reservation['log']) == reservation['log_sha256']
    queue = {}
    for name in ('queue020.json', 'queue021.json'):
        for row in read(PRIOR / name):
            queue.setdefault((row['op'], row['case_id']), row)
    todo = [{**row, 'prior_coverage_status': prior_by_key[key]['status']} for key, row in queue.items()]
    result = {'schema': 'tilelang-continuation-prior-audit/v1', 'worker': '7909c59978424284',
              'created_at': datetime.now(timezone.utc).isoformat(), 'delivery_source_hashes': delivered_sources,
              'ast_checks': ast_checks, 'attention_withdrawals': withdrawals,
              'bf16_core_byte_identical': source.read_bytes() == retained.read_bytes(),
              'artifact_checks': artifact_checks, 'receipts': receipts,
              'summary': {'prior_claimed_verified': len(receipts),
                          'full_output_receipts': sum(r['full_output_count_matches'] for r in receipts),
                          'native_levels': dict(Counter(r['native_evidence'] for r in receipts)),
                          'withdrawn_pass_receipts': sum(r['receipt_status'] == 'verified' for r in withdrawals),
                          'pending_deduplicated': len(todo)},
              'reservation020': reservation, 'immutable_evidence_hashes': frozen,
              'limits': ['Read-only audit is not root acceptance or new numerical verification.',
                         'Old source mismatches remain revision-bound; no old pass transferred to changed code.',
                         'Cache source/binary matching and successful JIT host calls do not prove direct driver load/launch identity.']}
    for name, data in [('prior_audit.json', result), ('todo_initial.json', todo)]:
        with (OUT / name).open('x') as stream:
            json.dump(data, stream, indent=2); stream.write('\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
