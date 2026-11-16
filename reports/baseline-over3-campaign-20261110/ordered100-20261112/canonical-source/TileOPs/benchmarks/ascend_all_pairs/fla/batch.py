"""Foreground, isolated case processes inside one bounded host reservation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from environment import configured_env

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--seconds",type=float,default=270)
    args=parser.parse_args()
    rows=json.loads(args.queue.read_text())
    started=time.monotonic()
    marker=EVIDENCE/'prepare_partial_native'
    if marker.exists():
        with (EVIDENCE/'prepare_partial_native.log').open('a') as log:
            prepared=subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_partial_native.py'))],stdout=log,stderr=subprocess.STDOUT)
        if prepared.returncode:return prepared.returncode
        marker.unlink()
    pending=[item for item in rows if not (EVIDENCE/'runs'/item['receipt']).exists()]
    groups=[]
    for item in pending:
        if not groups or groups[-1][0].get('mode','triton')!=item.get('mode','triton') or item.get('kind') in ('oracle_diagnostic','launcher_smoke'):
            groups.append([])
        groups[-1].append(item)
    for items in groups:
        item=items[0]
        output=EVIDENCE / "runs" / item["receipt"]
        if output.exists():
            continue
        if time.monotonic()-started>args.seconds-35:
            break
        mode=item.get("mode","triton")
        env=configured_env(mode)
        env["ASCEND_PROCESS_LOG_PATH"]=str(EVIDENCE/"native_logs"/output.stem)
        env["ASCEND_GLOBAL_LOG_LEVEL"]="0" if os.environ.get("FLA_ALLPAIRS_DEBUG_NATIVE")=="1" and mode!="triton" else "3"
        env["ASCEND_SLOG_PRINT_TO_STDOUT"]="0"
        group_queue=EVIDENCE/("active-group-"+output.stem+".json")
        group_queue.write_text(json.dumps(items,indent=2)+'\n')
        command=[sys.executable,str(Path(__file__).with_name("group.py")),str(group_queue),'--seconds',str(args.seconds-(time.monotonic()-started))]
        if item.get('kind')=='oracle_diagnostic':
            command=[sys.executable,str(Path(__file__).with_name('diagnose_case.py')),'--op',item['op'],'--case-id',item['case_id'],'--output',str(output)]
        if item.get('kind')=='launcher_smoke':
            command=[sys.executable,str(Path(__file__).with_name('launch_case.py')),'--op',item['op'],'--case-id',item['case_id'],'--output',str(output)]
        log=output.with_suffix(".group.log")
        active=dict(phase='running',started_at=datetime.now(timezone.utc).isoformat(),mode=mode,first=item,
                    case_count=len(items),group_queue=str(group_queue),log=str(log),command=command)
        (EVIDENCE/'active_group.json').write_text(json.dumps(active,indent=2)+'\n')
        with log.open("w") as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        active.update(phase='finished',finished_at=datetime.now(timezone.utc).isoformat(),returncode=result.returncode)
        (EVIDENCE/'active_group.json').write_text(json.dumps(active,indent=2)+'\n')
        complete=[entry for entry in items if (EVIDENCE/'runs'/entry['receipt']).exists()]
        last=complete[-1] if complete else item
        output=EVIDENCE/'runs'/last['receipt']
        summary=json.loads(output.read_text()) if output.exists() else {"status":"process_failed","returncode":result.returncode}
        if result.returncode:
            summary={"status":"process_failed","returncode":result.returncode,"error":"Checker group process exited unexpectedly; inspect active_case.json and group log."}
        print(json.dumps({"op":last["op"],"case_id":last["case_id"],"status":summary.get("status"),"error":summary.get("error"),"receipt":str(output)}),flush=True)
        (EVIDENCE/"batch_progress.json").write_text(json.dumps({"last":last,"status":summary.get("status"),"elapsed_seconds":time.monotonic()-started},indent=2)+"\n")
        if summary.get("status") in ("device_sync_failed", "process_failed"):
            break


if __name__=="__main__":
    raise SystemExit(main() or 0)
