"""CPU-only foreground coordinator for finite, guarded verification batches."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla'
CACHE=ROOT/'.cache/all-pairs/fla'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def pending(queue):
    rows=json.loads(queue.read_text())
    result=[]
    for row in rows:
        path=EVIDENCE/'runs'/row['receipt']
        if path.exists():
            recorded=json.loads(path.read_text())
            if (recorded.get('op'),recorded.get('case_id'))!=(row['op'],row['case_id']):
                raise RuntimeError('Receipt name belongs to a different manifest case: '+str(path))
        else:
            result.append(row)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('queue',type=Path)
    parser.add_argument('--max-batches',type=int,default=1000)
    args=parser.parse_args()
    queue=args.queue.resolve()
    logs=EVIDENCE/'guard_logs'
    logs.mkdir(exist_ok=True)
    for batch in range(args.max_batches):
        rows=pending(queue)
        if not rows:
            save(EVIDENCE/'continuation_status.json',dict(status='all_queued_cases_attempted',updated_at=now(),queue=str(queue)))
            return 0
        # This only coordinates safe source/deployment updates between batches;
        # it never interrupts a reservation or changes the host lock.
        if (EVIDENCE/'stop_after_batch').exists():
            save(EVIDENCE/'continuation_status.json',dict(status='stopped_between_batches',updated_at=now(),remaining=len(rows),queue=str(queue)))
            return 0
        started=now()
        stamp=started.replace(':','').replace('.','').replace('+','_')
        log=logs/('batch-'+stamp+'.log')
        command=['npu-guard','run','--owner','allpairs/fla','--wait','3600','--poll','0.1','--timeout','300','--',
                 'env','PYTHONDONTWRITEBYTECODE=1','TRITON_CACHE_DIR='+str(CACHE/'triton'),'TMPDIR='+str(CACHE/'tmp'),
                 str(ROOT/'.tools/baselines/baseline-run'),'flash-linear-attention-npu','python',
                 str(Path(__file__).with_name('batch.py')),str(queue)]
        save(EVIDENCE/'continuation_status.json',dict(status='waiting_or_running_guard',updated_at=started,batch=batch,remaining=len(rows),queue=str(queue),command=command,guard_log=str(log)))
        print(json.dumps(dict(event='guard_start',batch=batch,remaining=len(rows),log=str(log))),flush=True)
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        left=pending(queue)
        receipt=dict(started_at=started,finished_at=now(),guard_exit_code=result.returncode,command=command,
                     queue=str(queue),guard_log=str(log),before=len(rows),after=len(left))
        save(log.with_suffix('.json'),receipt)
        report_env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],
                       cwd=ROOT,env=report_env,check=True)
        print(json.dumps(dict(event='guard_end',exit_code=result.returncode,remaining=len(left),log=str(log))),flush=True)
        progress=EVIDENCE/'batch_progress.json'
        last=json.loads(progress.read_text()) if progress.exists() else {}
        stop_reason=None
        if result.returncode:
            stop_reason='guard_returned_'+str(result.returncode)
        elif last.get('status') in ('device_sync_failed','process_failed'):
            stop_reason=last['status']
        elif len(left)==len(rows):
            stop_reason='no_completed_case_in_batch'
        if stop_reason:
            # In particular, never turn 75/76/78 into an unguarded retry.
            # A 124 timeout also stops for inspection of the actual active case.
            save(EVIDENCE/'continuation_status.json',dict(status='needs_inspection',reason=stop_reason,updated_at=now(),remaining=len(left),guard_receipt=str(log.with_suffix('.json'))))
            return result.returncode or 2
    save(EVIDENCE/'continuation_status.json',dict(status='batch_limit_reached',updated_at=now(),remaining=len(pending(queue))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
