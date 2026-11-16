"""Publish only exactly accepted case IDs; CPU only, no new numerical acceptance."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
WS=HERE.parents[3]
EVIDENCE=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn'

def main():
    coverage=json.loads((EVIDENCE/'coverage.json').read_text());accepted={};pinned={}
    for row in coverage['cases']:
        if row['status'] in ('verified','verified_historical','verified_preserved'):
            accepted.setdefault(row['op'],[]).append(row['case_id'])
            if row['status']=='verified_preserved':pinned.setdefault(row['op'],[]).append(row['case_id'])
    (HERE/'accepted_cases.json').write_text(json.dumps(accepted,indent=2)+'\n')
    (HERE/'accepted_pinned_cases.json').write_text(json.dumps(pinned,indent=2)+'\n')
    print('Published',len(accepted),'operators;',sum(map(len,accepted.values())),'accepted cases')
if __name__=='__main__':main()
