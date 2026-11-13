"""Serial, two fresh reservations. Stop on ANY failure; never retry or expand."""
import argparse
import json
from pathlib import Path
import subprocess
import signal
import time
import sys
from pilot import PROTOCOL


def gate(output,outer):
    env=json.loads((output/'environment.json').read_text())
    if env.get('status')!='verified' or env.get('protocol')!=PROTOCOL or env.get('envID')!='round-candidate03':raise ValueError('isolated environment not verified')
    backend=env['environment']['modules']['torch._inductor.fx_passes.post_grad']
    if backend['sha256']!='0be87418b543e882f4e7351a21a0a9bcf578b33a8be271c81b0e861d6da853db':raise ValueError('patched backend mismatch')
    r=json.loads((output/'receipt.json').read_text());w=json.loads((outer/'watchdog-receipt.json').read_text())
    if r['exit_code']!=0 or w['exit_code']!=0 or r['failure'] or r['audit_failed']:raise ValueError('guard failed')
    if not r['cleanup']['reaped'] or r['cleanup']['errors'] or r['cleanup']['post_cleanup_proc'] is not None:raise ValueError('cleanup failed')
    if not r['lock_open_at_cleanup'] or not r['parent_lock_closed_after_run'] or w['remaining_live'] or w['errors']:raise ValueError('lock/watchdog failed')
    report=json.loads((output/'measurement/admission.json').read_text())
    if report['status']!='measured_pending_guard_and_environment_acceptance' or report['windows']!=15 or report['flush_bytes']!=1024**3:raise ValueError('measurement incomplete')
    for v in ['eager','compile']:
        check=report['timed_checks'][v];control=report['negative_controls'][v]
        if not check['checked_outputs'] or check['checked_outputs']!=check['total_outputs'] or not control['complete_output_zeroed'] or not control['rejected']:raise ValueError('checker/control failed')
        if report['measurement_graph_counters'][v]['before']!=report['measurement_graph_counters'][v]['after']:raise ValueError('unstable graph counters')
        if report['timings'][v]['offline_replay']!='15/15 passed':raise ValueError('runner replay failed')
    return 'local_gate_only_pending_independent_raw_audit'


def run_queue(command_factory,outputs,launch=subprocess.call,check=gate):
    results=[]
    for index,(output,outer) in enumerate(outputs):
        rc=launch(command_factory(index,output,outer))
        row={'index':index,'exit':rc};results.append(row)
        if rc:return results,rc
        try:row['gate']=check(output,outer)
        except Exception as exc:row['error']=repr(exc);return results,78
    return results,0


def launch_owned(command,received,receipt,deadline_seconds=1270):
    child=subprocess.Popen(command,start_new_session=True)
    receipt.update(pid=child.pid,reaped=False)
    started=time.monotonic();sent=False
    while child.poll() is None:
        if (received or time.monotonic()-started>=deadline_seconds) and not sent:
            child.send_signal(received[0] if received else signal.SIGTERM)
            sent=True;receipt['stop_requested_at']=time.monotonic()
        if sent and time.monotonic()-receipt['stop_requested_at']>15:
            receipt['error']='watchdog did not exit after stop request; no next case'
            return 78  # Explicit survivor, never falsely claim cleanup or start next.
        time.sleep(.05)
    receipt.update(reaped=True,returncode=child.returncode)
    return 128+received[0] if received else (child.returncode if child.returncode>=0 else 128-child.returncode)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    base=Path(__file__).resolve().parent
    def command(i,out,outer):
        return [sys.executable,str(base/'case_watchdog.py'),'--output',str(outer),'--',sys.executable,str(base/'pilot.py'),'--root',str(a.root),'--output',str(out),'--index',str(i),'--manifest-sha',a.manifest_sha]
    received=[];handlers={};launches=[];results=[];code=78
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
        handlers[sig]=signal.signal(sig,lambda n,f:received.append(n))
    def launch(cmd):
        if received:return 128+received[0]
        record={};launches.append(record)
        return launch_owned(cmd,received,record)
    try:
        results,code=run_queue(command,[(a.output/f'case{i}',a.output/f'case{i}-outer') for i in range(4)],launch=launch)
    finally:
        (a.output/'controller-receipt.json').write_text(json.dumps({'effective_protocol':PROTOCOL,'results':results,'launches':launches,'signals':received,'exit':code,'acceptance':'alternate/pending'},indent=2))
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return code
if __name__=='__main__':raise SystemExit(main())
