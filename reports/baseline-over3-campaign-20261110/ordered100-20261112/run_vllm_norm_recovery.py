"""Serialized one-case full original API alternative; no successful original timing to compare."""
import pathlib,sys,fcntl,time,json,subprocess,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
f=P/'VLLM_NORM_RECOVERY_PROCESS.json';f.write_text(json.dumps({'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':[1178],'status':'waiting_serialized_controller_slot'},indent=2))
label='rollout-vllm-norm-candidate-1178'
while True:
 if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('STOP')
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-vllm','vllm-norm-candidate',[1178],label)
 if event is not None:break
 time.sleep(30)
if 'batch' in event:
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
 result=event['batch']['results'][0]
 summary={'status':result['status'],'result':result,'evidence':label+'.CHECKPOINT.json','original_current_timing_valid':False,'speedup_claimed':False,'kernel_checker_precision_unchanged':True,'not_added_to_original_ledger_automatically':True}
else:summary={'status':'guard_or_process_failed','evidence':label+'.FAILED.json','no_automatic_retry':True}
(P/'VLLM_NORM_API_RECOVERY_RESULT.json').write_text(json.dumps(summary,indent=2));f.write_text(json.dumps({'pid':os.getpid(),'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed_one_case','result':summary['status']},indent=2));print(summary['status'],label,flush=True)
