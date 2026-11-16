"""Safe foreground handoff; resource-invalid reductions get independent bounded guards."""
import pathlib,json,fcntl,os,signal,time,datetime,sys
P=pathlib.Path(__file__).resolve().parent;pid=json.loads((P/'CURRENT_PROCESS.json').read_text())['pid'];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def put(n,v):(P/n).write_text(json.dumps(v,indent=2))
put('REDUCTION_RESOURCE_HANDOFF.json',{'status':'waiting_active_guard_and_archive','previous_pid':pid,'pid':os.getpid()})
with (P/'controller.lock').open('a') as slot:
 fcntl.flock(slot,fcntl.LOCK_EX)
 for _ in range(12000):
  try:w=pathlib.Path('/proc/'+str(pid)+'/wchan').read_text()
  except FileNotFoundError:break
  if 'lock' in w:os.kill(pid,signal.SIGINT);break
  time.sleep(.1)
 else:raise RuntimeError('No safe controller boundary')
 for _ in range(100):
  if not pathlib.Path('/proc/'+str(pid)).exists():break
  time.sleep(.1)
 else:raise RuntimeError('Previous controller alive')
 f=P/'continue_ordered.py';b=f.read_bytes();f.with_name(f.name+'.before-reduction-resource-'+stamp).write_bytes(b);s=b.decode();old="batch_limit=1 if row['library'] in ('ops-nn','ops-math','ops-transformer','ops-tensor') else 4";new="batch_limit=1 if row['library'] in ('ops-nn','ops-math','ops-transformer','ops-tensor') or (row['library']=='tilelang-ascend' and row['op'] in ('AmaxFwdOp','AminFwdOp')) else 4";assert s.count(old)==1;s=s.replace(old,new);s=s.replace("('-cache-evidence2' if row['library']=='tilelang-ascend' else '')","('-resource-single1' if row['library']=='tilelang-ascend' and row['op'] in ('AmaxFwdOp','AminFwdOp') else '-cache-evidence2' if row['library']=='tilelang-ascend' else '')")
 s=s.replace("if any(s in error for s in ['Live foreground guard identity mismatch'", "if any(s in error for s in ['Unresolved call Op(tl.', 'Live foreground guard identity mismatch'")
 compile(s,str(f),'exec');f.write_text(s)
 f=P/'ROLLOUT_LEDGER.json';b=f.read_bytes();f.with_name(f.name+'.before-reduction-resource-'+stamp).write_bytes(b);j=json.loads(b);removed=[]
 for i in range(1639,1647):
  v=j.get(str(i));assert v and v['result']['status']=='guard_or_process_invalid_pending'
  assert v['result']['evidence'] in ('rollout-1639-1642-cache-evidence2.FAILED.json','rollout-1643-1646-cache-evidence2.FAILED.json')
  removed.append(v);del j[str(i)]
 f.write_text(json.dumps(j,indent=2));put('REDUCTION_RESOURCE_REQUEUED.json',{'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entries':removed,'change':'four cases/process -> one case/process, same600s guard, freshhealth mandatory; no kernel/math/checker edits','old_health_interrupted_and_timeout_archives_retained':True,'no_valid_rows_removed':True})
 put('CURRENT_PROCESS.json',{'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entry':'continue_ordered.py','foreground_tool_session':'see REDUCTION_RESOURCE_TOOL.json','own_migration_disabled':True})
put('REDUCTION_RESOURCE_HANDOFF.json',{'status':'ordered_controller_resumed_single_reduction_guard','previous_pid':pid,'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
