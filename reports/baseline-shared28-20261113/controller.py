"""Finite foreground FIFO-guarded run; never resumes another campaign."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

P=Path(__file__).resolve().parent
plan=json.loads((P/'PLAN.json').read_text());tasks=plan['scope']
assert len(tasks)==28
assert json.loads((P/'TOOLCHAIN.json').read_text())['status']=='passed'
for d in ['runs-isolated','logs-isolated','spool']: (P/d).mkdir(exist_ok=True)


def write(name,data):
    f=P/name;q=f.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2));q.replace(f)


def volume(t):
    sizes={'float32':4,'bfloat16':2,'float16':2,'int32':4,'int64':8}
    return sum(math.prod(v[0])*sizes.get(v[1],4) for v in t['case']['tensors'].values() if isinstance(v,list) and len(v)==2 and isinstance(v[0],list))


def prepare(t):
    run=P/'runs-isolated'/t['id'];run.mkdir(exist_ok=True)
    if (run/'PREPARED.json').exists():return None
    log=(P/'logs-isolated'/f"{t['id']}-prepare.log").open('a')
    proc=subprocess.Popen(['bash',str(P/'launch.sh'),'prepare',str(P/'tasks'/f"{t['id']}.json"),str(run)],stdout=log,stderr=subprocess.STDOUT)
    return proc,log,time.monotonic()


accepted=[];failed=[];prep=None
write('STATE.json',dict(status='running',accepted=accepted,failed=failed,total=28,phase='CPU preparation'))
for index,t in enumerate(tasks):
    run=P/'runs-isolated'/t['id']
    if (run/'ACCEPTED.json').exists():
        accepted.append(json.loads((run/'ACCEPTED.json').read_text()));continue
    if prep is None:prep=prepare(t)
    if prep:
        proc,log,when=prep
        try:rc=proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.terminate();proc.wait();rc=124
        log.close();prep=None
        if rc:
            failed.append(dict(id=t['id'],key=t['key'],phase='CPU preparation',exit_code=rc))
            write('STATE.json',dict(status='failed_needs_diagnosis',accepted=accepted,failed=failed,total=28));raise SystemExit(rc)
    assert (run/'PREPARED.json').exists()
    write('STATE.json',dict(status='running',accepted=accepted,failed=failed,total=28,current=t['key'],phase='queued_or_device',owner='root/shared28-isolated/'+t['id']))
    command=['/usr/local/bin/npu-guard','run','--owner','root/shared28-isolated/'+t['id'],'--wait','21600','--timeout','3600','--receipt',str(run/'guard.json'),'--','bash',str(P/'launch.sh'),'device',str(P/'tasks'/f"{t['id']}.json"),str(run)]
    with (P/'logs-isolated'/f"{t['id']}-device.log").open('a') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        write('CURRENT_PROCESS.json',dict(controller_pid=__import__('os').getpid(),guard_pid=proc.pid,owner='root/shared28-isolated/'+t['id'],key=t['key'],command=command))
        # Known CPU-only preparation overlaps queued/device work, with bounded host memory.
        if index+1<len(tasks) and volume(t)+volume(tasks[index+1])<32*1024**3:
            prep=prepare(tasks[index+1])
        rc=proc.wait()
    if rc:
        if prep:
            prep[0].wait();prep[1].close();prep=None
        failed.append(dict(id=t['id'],key=t['key'],phase='guard/device',exit_code=rc))
        write('STATE.json',dict(status='failed_needs_diagnosis',accepted=accepted,failed=failed,total=28));raise SystemExit(rc)
    g=json.loads((run/'guard.json').read_text());m=g.get('metadata',g)
    assert g['phase']=='finished' and g['workload_exit_code']==0
    assert m['performance_valid'] and not m.get('conflict_detected') and not m.get('visibility_lost')
    measurement=json.loads((run/'measurement.json').read_text());correct=json.loads((run/'correctness.json').read_text())
    assert correct['check']['checked_outputs']==correct['check']['total_outputs']>0
    assert len(measurement['timing']['latency_samples_ms'])==15
    assert measurement['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
    row=dict(id=t['id'],key=t['key'],device_us=measurement['timing']['latency_ms']*1000,
             protocol=measurement['timing']['measurement_protocol'],previous_measurement_id=t['previous_measurement_id'],
             measurement_id=hashlib.sha256((run/'measurement.json').read_bytes()).hexdigest(),
             source_sha256=measurement['source_sha256'],guard_sha256=hashlib.sha256((run/'guard.json').read_bytes()).hexdigest(),
             correct_outputs=correct['check']['checked_outputs'],raw_replayed_windows=15,guard_valid=True)
    (run/'ACCEPTED.json').write_text(json.dumps(row,indent=2));accepted.append(row)
    print(json.dumps(dict(progress=len(accepted),total=28,accepted=row)),flush=True)
    write('RESULTS.json',accepted)
    write('STATE.json',dict(status='running',accepted=accepted,failed=failed,total=28,phase='next_case'))
write('STATE.json',dict(status='complete',accepted=accepted,failed=failed,total=28,phase='all_measured',publication='not_published'))
print('ALL28_COMPLETE',flush=True)
