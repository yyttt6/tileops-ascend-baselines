"""Reuse runtime imports for a homogeneous, foreground group of original checks."""
import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from check import main as check_case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('queue',type=Path)
    parser.add_argument('--seconds',type=float,default=250)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[4]
    evidence=root/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla'
    started=time.monotonic()
    for item in json.loads(args.queue.read_text()):
        output=evidence/'runs'/item['receipt']
        if output.exists():continue
        if time.monotonic()-started>args.seconds-20:break
        sys.argv=[__file__,'--op',item['op'],'--case-id',item['case_id'],'--library','flash-linear-attention-npu','--output',str(output)]
        (evidence/'active_case.json').write_text(json.dumps({'phase':'running','case':item,'started_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
        check_case()
        row=json.loads(output.read_text())
        (evidence/'active_case.json').write_text(json.dumps({'phase':'finished','case':item,'finished_at':datetime.now(timezone.utc).isoformat(),'status':row['status']},indent=2)+'\n')
        print(json.dumps({'op':item['op'],'case_id':item['case_id'],'status':row['status'],'error':row.get('error')}),flush=True)
        (evidence/'batch_progress.json').write_text(json.dumps({'last':item,'status':row['status'],'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
        if row['status']=='device_sync_failed':break
        gc.collect()


if __name__=='__main__':main()
