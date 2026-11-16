from pathlib import Path
import json,subprocess,sys,datetime,os
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_sequential_family_handoff','scope':[761,763,778,782]}
def put():
 state['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=A/'BF16_ORIGINAL_CONTRACT_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
try:
 for group in ['bmm','instance']:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists() and json.loads((D.parent/'STATE.json').read_text())['status']=='active'
  state.update(status='child_original_native_queue',family=group);put()
  assert subprocess.run([sys.executable,str(D/('queue_ops_transformer_'+group+'_bf16_original_contract.py'))]).returncode==0,'Family failed; diagnose before any continuation'
 state['status']='both_families_completed_next_group_required'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:put()
