"""CPU test subprocess only; never imports device driver."""
import json
import os
import signal
import sys
import time
import resident

mode, r, w, session = sys.argv[1:]
r, w = int(r), int(w)
g, _ = resident.original()
runtime = mode.startswith('runtime_')
if runtime: mode=mode.removeprefix('runtime_')
base_mode=mode
count=0
def sample():
    global count, mode
    count += 1
    mode = 'ok' if runtime and count == 1 else base_mode
    s = dict(time=g.now(), device=0, errors=[], utilization={k: {'rc':0,'value':0} for k in ['aicore','hbm_bw','vector','npu','cube']},
             pid_query_rc=0,pids=[],health={'rc':0,'value':0},memory={'rc':0},core_frequency=[],power={'rc':0},
             fd_candidates=[],unreadable_proc_pids=[],observation_protocol='reservation-driver-v1-diagnostic04',
             fd_scan_performed=False,observation_limit='Unregistered users opening device FDs are not observed')
    if mode == 'unhealthy': s['health']['value']=1
    if mode == 'pid_failure': s['pid_query_rc']=1
    if mode == 'owned':
        from pathlib import Path
        meta=json.loads(Path(os.environ['CPU_LOCK_PATH']).read_text())
        if meta.get('child'):s['pids']=[g.proc_identity(meta['child']['pid'])]
    if mode == 'foreign': s['pids']=[g.proc_identity(os.getppid())]
    if mode == 'busy': s['utilization']['npu']['value']=90
    if mode == 'query_hang': time.sleep(3)
    if mode == 'term_ignored': signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3)
    if mode == 'corrupt_sample': return {}
    if mode == 'death': os._exit(2)
    s['state'],s['reasons']=g.classify(s)
    if mode == 'missing': del s['utilization']['cube']
    return s
if mode == 'load_ignoreterm':
    signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3)
if mode == 'load_hang': time.sleep(3)
if mode in ['ok','unhealthy','pid_failure','foreign','busy','query_hang','term_ignored','death','missing','corrupt_sample','owned']:
    resident.worker(r,w,session,lambda:sample)
else:
    common=dict(version=resident.VERSION,session=session,pid=os.getpid(),ticks=resident.ticks(os.getpid()),pgid=os.getpgrp())
    resident.send(w,dict(common,kind='ready'))
    with os.fdopen(r,'rb') as inp:
        req=json.loads(inp.readline())
        a=time.monotonic(); s=sample()
        response=dict(common,kind='sample',id=req['id'],start=a,end=time.monotonic(),sample=s)
        if mode=='old':response['start']=0
        if mode=='sequence':response['id']+=1
        if mode=='identity':response['ticks']+=1
        if mode=='session':response['session']='other'
        if mode=='corrupt':os.write(w,b'not json\n')
        elif mode=='partial':os.write(w,b'{');time.sleep(3)
        elif mode=='duplicate':
            data=(json.dumps(response)+'\n').encode();os.write(w,data+data)
        else:resident.send(w,response)
        time.sleep(3)
