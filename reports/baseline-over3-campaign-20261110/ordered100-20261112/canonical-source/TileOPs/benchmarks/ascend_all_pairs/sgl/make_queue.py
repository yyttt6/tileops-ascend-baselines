"""Plan only real inventory cases; numeric status is never inferred here."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl'
sys.path.insert(0,str(ROOT/'TileOPs'))
from benchmarks.ascend_all_pairs.sgl import PROVIDERS
from benchmarks.ascend_all_pairs.sgl.eligibility import restriction


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ops',nargs='*')
    p.add_argument('--statuses',nargs='+',default=['supported_untested','historical_source_drift','historical_failure','verified_prior_adapter_revision'])
    a=p.parse_args()
    output=a.output.resolve()
    if not output.is_relative_to(EVIDENCE):raise ValueError('Queue output must remain in this worker evidence directory')
    inv=json.loads((EVIDENCE/'inventory_snapshot_v2.json').read_text())
    bykey={(r['op'],r['case_id']):r for r in inv['cases']}
    coverage=json.loads((EVIDENCE/'coverage.json').read_text())
    chosen=[]
    for row in coverage['cases']:
        op=row['op']
        if op not in PROVIDERS or row['status'] not in a.statuses or a.ops and op not in a.ops:continue
        spec=bykey[op,row['case_id']]
        if restriction(spec):continue
        chosen.append(spec)
    # Small complete workloads go first; attention/recurrence long cases still
    # remain in the queue and cannot be reclassified by this planning estimate.
    chosen.sort(key=lambda c:(sum(math.prod(t[0]) for t in c['tensors'].values()),c['op'],c['case_id']))
    queue=[{'op':c['op'],'case_id':c['case_id']} for c in chosen]
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(queue,indent=2)+'\n')
    print(json.dumps({'output':str(output),'cases':len(queue),'ops':len({c['op'] for c in queue})}))

if __name__=='__main__':main()
