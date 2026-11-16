#!/usr/bin/env python3
"""Publish only cases whose complete numerical and native receipts are accepted."""
import json
from pathlib import Path
from specs import EVIDENCE

coverage = json.loads((EVIDENCE / 'coverage.json').read_text())
cases = {
    row['op']: [c['case_id'] for c in row['cases'] if c['status'] == 'verified']
    for row in coverage['operators']
}
cases = {op: ids for op, ids in cases.items() if ids}
own = Path(__file__).parent
(own / 'enabled_cases.json').write_text(json.dumps(cases, indent=2) + '\n')
(own / 'enabled.json').write_text(json.dumps(sorted(cases), indent=2) + '\n')
print(json.dumps({'published_operators': len(cases),
                  'published_cases': sum(map(len, cases.values()))}))
