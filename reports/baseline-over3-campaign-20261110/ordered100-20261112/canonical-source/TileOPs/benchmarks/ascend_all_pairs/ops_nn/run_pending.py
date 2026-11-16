"""Foreground guarded batches of unattempted exact manifest cases, CPU controller."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
WS=HERE.parents[3]
EVIDENCE=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn'
CACHE=WS/'.cache/all-pairs/ops_nn'

def refresh():
    for name in ('recover_native.py','audit.py','report.py','publish.py'):
        subprocess.run([sys.executable,str(HERE/name)],cwd=WS,check=True,stdout=subprocess.DEVNULL)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--max-batches',type=int,default=100)
    p.add_argument('--poll',type=float,default=1.0);p.add_argument('--batch-size',type=int,default=100);p.add_argument('--install',default='install3');args=p.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    env=os.environ.copy();env['OPS_NN_VENDOR']=str(CACHE/args.install/'vendors/allpairs_nn')
    for number in range(args.max_batches):
        if (EVIDENCE/(args.prefix+'.STOP')).exists():
            print(json.dumps({'stopped_at_guard_boundary':True,'prefix':args.prefix}),flush=True);return 0
        refresh();coverage=json.loads((EVIDENCE/'coverage.json').read_text())
        # A numerical mismatch is case-specific: do not retry that unchanged case,
        # but continue other manifest shapes/dtypes of the same source API.
        blocked={r['op'] for r in coverage['cases'] if r['status'] in ('deployment_blocked','device_failed','numerical_pass_attribution_pending')}
        pending=[r for r in coverage['cases'] if r['status'] in ('supported_unverified','needs_revalidation') and r['op'] not in blocked]
        if not pending:
            print(json.dumps({'no_eligible_unattempted_cases':True,'operators_requiring_failure_review':sorted(blocked)}),flush=True);return 0
        attempted={r['op'] for r in coverage['cases'] if r.get('trials')}
        def size(r):
            return sum(math.prod(v[0]) for v in r['params'].values() if isinstance(v,list) and len(v)==2 and isinstance(v[0],list) and isinstance(v[1],str))
        # First cover each untouched API with one actual manifest case, then expand cases.
        pending.sort(key=lambda r:(r['op'] in attempted,size(r),r['op'],r['case_id']))
        selected=[];seen=set()
        for row in pending:
            if row['op'] not in attempted and row['op'] in seen:continue
            selected.append(dict(op=row['op'],case_id=row['case_id']));seen.add(row['op'])
            if len(selected)>=args.batch_size:break
        tag=f'{args.prefix}-{number:03d}';out=EVIDENCE/tag;jobs=EVIDENCE/(tag+'-jobs.json');log=CACHE/(tag+'.log')
        if out.exists() or jobs.exists() or log.exists():raise FileExistsError(tag)
        jobs.write_text(json.dumps(selected,indent=2)+'\n')
        cmd=['npu-guard','run','--owner','allpairs/ops_nn','--wait','3600','--poll',str(args.poll),'--timeout','900','--',
             'bash',str(HERE/'run_batch.sh'),'--jobs',str(jobs),'--out',str(out),'--seconds','800']
        event=dict(started_at=datetime.now(timezone.utc).isoformat(),command=cmd,jobs=str(jobs.relative_to(WS)),log=str(log.relative_to(WS)),install=args.install)
        print(json.dumps(event),flush=True)
        with log.open('w') as stream:
            result=subprocess.run(cmd,cwd=WS,env=env,stdout=stream,stderr=subprocess.STDOUT)
        event.update(exit_code=result.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        (EVIDENCE/(tag+'-guard.json')).write_text(json.dumps(event,indent=2)+'\n');refresh()
        print(json.dumps(event),flush=True)
        if result.returncode!=0:return result.returncode
    return 0
if __name__=='__main__':raise SystemExit(main())
