"""Correct historical op-level exclusions without overwriting the source ledger."""
import argparse
import hashlib
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = args.inventory.read_bytes()
    source = json.loads(raw)
    rows = []
    for row in source:
        if row['category'] != 'outside_current_specialist_scope_provisional':
            continue
        rows.append({
            'contract_sha256': row['contract_sha256'], 'op': row['op'], 'case': row['case'],
            'previous_category': row['category'], 'current_category': 'scope_pending_exact_contract_review',
            'excluded': False, 'receiving_owner_verified': False,
            'prior_evidence': row.get('prior_compile_candidates', []),
            'resolution_requires': 'Exact-contract existing coverage, specific quant/anomaly-task exclusion, or explicit receiving ownership; op name alone is insufficient.',
        })
    assert len(rows) == len({row['contract_sha256'] for row in rows}) == 357
    result = {'source_inventory_sha256': hashlib.sha256(raw).hexdigest(),
              'review': 'https://github.com/yyttt6/agent-hub/issues/3#issuecomment-6853646214',
              'count': len(rows), 'final_exclusions': 0, 'rows': rows}
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('357 unique contracts marked scope pending; zero final exclusions')
