"""Original checker, one foreground reservation, complete-case soft boundary."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from environment import ROOT, configured_env

EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla_continuation'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('queue',type=Path)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--seconds',type=float,default=60)
    args=parser.parse_args()
    rows=json.loads(args.queue.read_text())
    if not args.child:
        from check import checker
        checker.require_guard()
        assert len({x['mode'] for x in rows})==1
        env=configured_env(rows[0]['mode'])
        env['FLA_CONTINUATION_EVIDENCE']=str(EVIDENCE)
        env['ASCEND_PROCESS_LOG_PATH']=str(EVIDENCE/'native_logs'/args.queue.stem)
        command=[sys.executable,'-B',__file__,str(args.queue),'--child','--seconds',str(args.seconds)]
        return subprocess.run(command,env=env).returncode
    from check import main as check_case
    started=time.monotonic()
    for index,item in enumerate(rows):
        if index and time.monotonic()-started>=args.seconds:break
        output=EVIDENCE/'runs'/item['receipt']
        if output.exists():raise FileExistsError(output)
        sys.argv=[__file__,'--op',item['op'],'--case-id',item['case_id'],'--library','flash-linear-attention-npu','--output',str(output)]
        active=dict(phase='running',case=item,started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),pgid=os.getpgrp())
        (EVIDENCE/'active_case.json').write_text(json.dumps(active,indent=2)+'\n')
        rc=check_case()
        row=json.loads(output.read_text())
        active.update(phase='finished',status=row['status'],checker_returncode=rc,finished_at=datetime.now(timezone.utc).isoformat())
        (EVIDENCE/'active_case.json').write_text(json.dumps(active,indent=2)+'\n')
        if rc==78 or row['status'] in ('device_sync_failed','cleanup_failed'):return rc or 1
        gc.collect()
    return 0


if __name__=='__main__':raise SystemExit(main())
