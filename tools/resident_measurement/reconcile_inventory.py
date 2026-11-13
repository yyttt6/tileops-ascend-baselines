"""CPU-only evidence classification. No workload imports or device operations."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def reconcile(checkout):
    commission = checkout/'commissions/torch-compile-20261110'
    inventory = json.loads((commission/'CASE_INVENTORY.json').read_text())
    prior = json.loads((commission/'PRIOR_COMPILE_RESULTS.json').read_text())
    declarations = defaultdict(list)
    for path in sorted((checkout/'reports/torch-v010-20261106/source/benchmarks/_cases').glob('*.py')):
        raw = path.read_bytes()
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith('Op'):
                declarations[node.value].append({'file': str(path.relative_to(checkout)), 'line': node.lineno,
                                                'sha256': hashlib.sha256(raw).hexdigest()})
    lookup = {(r['op'], r['case_id']): r for r in inventory['rows']}
    matches = defaultdict(list)
    unmatched = []
    counts = Counter()
    for dataset in ('quant', 'high_speedup'):
        for ordinal, item in enumerate(prior[dataset]):
            key = (item['op'], item['case'])
            row = lookup.get(key)
            record = {'dataset': dataset, 'ordinal': ordinal, 'op': key[0], 'case': key[1],
                      'variant': item.get('variant'), 'recorded_status': item.get('status'),
                      'recorded_windows': item.get('windows'), 'result_sha256': item.get('result_sha256'),
                      'same_environment_protocol_verified': False}
            if row is None:
                record['classification'] = 'outside_frozen_inventory'; unmatched.append(record); continue
            if dataset == 'quant':
                equal = item.get('parameters') == row['case']['params']
                record['parameters_equal'] = equal
                record['classification'] = 'historical_quant_parameters_match' if equal else 'historical_quant_parameters_differ'
            elif item.get('status') == 'failed':
                record['classification'] = 'historical_compile_failed'
            elif item.get('windows') == 1:
                record['classification'] = 'historical_one_window_pilot'
            elif item.get('windows') == 15 and item.get('status') == 'formal_raw_verified':
                record['classification'] = 'historical_formal_raw_verified_separate_protocol'
            else:
                record['classification'] = 'historical_unclassified'
            counts[record['classification']] += 1
            matches[row['contract_sha256']].append(record)
    rows = [{'contract_sha256': r['contract_sha256'], 'op': r['op'], 'case': r['case_id'],
             'prior_evidence': matches[r['contract_sha256']],
             'factory_name_declarations': declarations[r['op']],
             'implementation_status': 'factory_name_declared_runtime_callable_not_evaluated' if declarations[r['op']] else 'factory_name_not_found_requires_review'}
            for r in inventory['rows']]
    return {'summary': {'inventory_contracts': len(rows), 'unique_ops': len({r['op'] for r in rows}),
                        'contracts_with_prior_evidence': sum(bool(r['prior_evidence']) for r in rows),
                        'prior_record_classifications': dict(counts), 'prior_records_outside_inventory': len(unmatched),
                        'contracts_with_factory_name_declarations': sum(bool(r['factory_name_declarations']) for r in rows),
                        'verified_same_environment_protocol_prior_coverage': 0,
                        'limitations': ['Static factory-name declarations are not a callable/runtime implementation test.',
                                        'Prior parameter equality does not establish source or environment equivalence.',
                                        'Historical failed/pilot/formal records remain separate; do not count all name matches as coverage.']},
            'rows': rows, 'prior_records_outside_inventory': unmatched}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.checkout.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], indent=2))
