import pathlib,json,fcntl,os,signal,time,datetime,sys
P=pathlib.Path(__file__).resolve().parent;pid=json.loads((P/'CURRENT_PROCESS.json').read_text())['pid'];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def put(n,v):(P/n).write_text(json.dumps(v,indent=2))
put('COMMON_RUNTIME_HANDOFF.json',{'status':'waiting_active_guard_and_archive','previous_pid':pid,'pid':os.getpid()})
with (P/'controller.lock').open('a') as slot:
 fcntl.flock(slot,fcntl.LOCK_EX)
 for _ in range(12000):
  try:w=pathlib.Path('/proc/'+str(pid)+'/wchan').read_text()
  except FileNotFoundError:break
  if 'lock' in w:os.kill(pid,signal.SIGINT);break
  time.sleep(.1)
 else:raise RuntimeError('No safe boundary')
 for _ in range(100):
  if not pathlib.Path('/proc/'+str(pid)).exists():break
  time.sleep(.1)
 else:raise RuntimeError('Old controller alive')
 f=P/'continue_ordered.py';b=f.read_bytes();f.with_name(f.name+'.before-common-runtime-'+stamp).write_bytes(b);s=b.decode();old="['Unresolved call Op(tl.', 'Live foreground guard identity mismatch'";new="['Tensor out not implemented for DT_UINT8', \"Select this library's OPP with baseline-run\", 'Load the isolated original vendor before importing Torch.', 'Unresolved call Op(tl.', 'Live foreground guard identity mismatch'";assert s.count(old)==1;s=s.replace(old,new);s=s.replace('unresolved pinned absolute-path registry manifest; no further NPU trials before sourcebundle repair','CPU registry import repaired; original native OPP and infer shim absent; source-bound native profile required before further device trials');compile(s,str(f),'exec');f.write_text(s)
 put('CURRENT_PROCESS.json',{'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entry':'continue_ordered.py','foreground_tool_session':'see COMMON_RUNTIME_TOOL.json','own_migration_disabled':True})
put('COMMON_RUNTIME_HANDOFF.json',{'status':'resumed_ordered_no_repeated_known_runtime_fault_per_operator','previous_pid':pid,'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
