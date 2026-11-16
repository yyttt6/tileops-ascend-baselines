"""One agent's serialized candidate follow-up; never overlaps main controller's guard."""
import pathlib,sys,json,time,fcntl,subprocess,os,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'sgl-candidate-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=json.loads((P/'TASKS.json').read_text());path=P/'SGL_WRAPPER_CANDIDATES.json';done=json.loads(path.read_text()) if path.exists() else {};ops={'LerpScalarFwdOp','NegFwdOp','SiluAndMulFwdOp'}
def put(f,x):
 tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(f)
put(P/'SGL_CANDIDATE_MONITOR_PROCESS.json',{'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'same_controller_lock':'controller.lock','physical_guard':'same opportunistic host guard, failfast, never parallelNPU','current_scope':sorted(ops),'select_candidate_requires_observed_device_gain':0.03})
while True:
 if (P/'STOP').exists() or (P.parent/'STOP').exists():break
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());available=[]
 for r in rows:
  i=r['ordinal']
  if r['library']=='sgl-kernel-npu' and r['op'] in ops and str(i) not in done and ledger.get(str(i),{}).get('result',{}).get('status')=='raw_checked_pending_guard':available.append(i)
 if not available:time.sleep(20);continue
 i=available[0];key=str(i);base=ledger[key];label='rollout-sgl-wrapper-candidate-'+str(i)
 with (P/'controller.lock').open('a') as slot:
  try:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:time.sleep(5);continue
  event=c.run('r-sgl','sgl-candidate',[i],label)
 if event is None:time.sleep(30);continue
 record={'ordinal':i,'label':label,'original_evidence':base['label']+'.CHECKPOINT.json','selection':'original','status':'pending_guard_or_process_failure'}
 if 'batch' in event:
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);result=event['batch']['results'][0];record['result']=result;record['status']=result['status']
  if result['status']=='raw_checked_pending_guard':
   original=base['result'];old=json.loads((P/(base['label']+'.CHECKPOINT.json')).read_text());assert old['batch']['runtime']==event['batch']['runtime'];assert original['payload_sha256']==result['payload_sha256'] and original['prepare_sha256']==result['prepare_sha256'];assert all(result['sources'].get(k)==v for k,v in original['sources'].items());assert original['metric']==result['metric']=='device_async'
   gain=1-result['device_us']/original['device_us'];record.update(device_gain_fraction=gain,original_device_us=original['device_us'],candidate_device_us=result['device_us'],candidate_selected=(gain>=0.03),selection='reviewed_wrapper_candidate' if gain>=0.03 else 'original',source_input_runtime_matched=True)
 else:record['failure_evidence']=label+'.FAILED.json'
 done[key]=record;put(path,done);print('CANDIDATE_REVIEWED',i,record['status'],record['selection'],record.get('device_gain_fraction'),flush=True);time.sleep(2)
