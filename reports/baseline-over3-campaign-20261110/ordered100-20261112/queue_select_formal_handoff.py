from pathlib import Path
import json,sys,os,time,fcntl,datetime,subprocess
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113'
lock=(P/'select-formal-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(x):
 p=P/'SELECT_FORMAL_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
s={'entry':Path(__file__).name,'pid':os.getpid(),'status':'wait_real_Round20_and_H1_Select6_probe','queued_at':now(),'not_primary_until_exec':True};put(s)
while True:
 assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
 q=json.loads((A/'SERVER1_SELECT_ATTRIBUTION.json').read_text());c=json.loads((P/'OPS_NN_ROUND_EXPANSION.json').read_text());inputs=json.loads((P/'OPS_NN_SELECT_INPUTS.json').read_text())
 assert q['status'] not in ('functional_probe_failed_requires_diagnosis','stopped_requires_diagnosis'),q
 assert c['status']!='stopped_requires_diagnosis',c
 assert inputs['status']!='CPU_source_input_fault_requires_diagnosis',inputs
 if c['accepted_original_APIs']==20 and q['status']=='functional_probe_passed_not_formal_baseline' and inputs.get('verified_inputs')==28:break
 time.sleep(10)
subprocess.run([sys.executable,str(P/'qualify_select_private_cpu.py')],check=True)
s.update(status='exec_qualified28_original_API_formal',checked_at=now());put(s)
os.execv(sys.executable,[sys.executable,'-u',str(P/'queue_ops_nn_select.py')])
