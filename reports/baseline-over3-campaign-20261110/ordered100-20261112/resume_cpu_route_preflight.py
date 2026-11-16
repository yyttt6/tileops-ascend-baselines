import pathlib,json,fcntl,os,signal,time,datetime,sys,hashlib
P=pathlib.Path(__file__).resolve().parent;pid=json.loads((P/'CURRENT_PROCESS.json').read_text())['pid'];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def put(n,v):(P/n).write_text(json.dumps(v,indent=2))
put('CPU_ROUTE_HANDOFF.json',{'status':'waiting_active_guard_and_archive','previous_pid':pid,'pid':os.getpid()})
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
 f=P/'continue_ordered.py';b=f.read_bytes();f.with_name(f.name+'.before-CPU-route-'+stamp).write_bytes(b);s=b.decode();needle='i=100\nwhile i<len(rows):';new='''route_catalog=json.loads((P/'CPU_TLA_ROUTE_CATALOG.json').read_text())
assert route_catalog['NoNPU'] and route_catalog['classic_lookup_AST_proved_op_only']
for name,digest in route_catalog['sources'].items():
 assert c.sha(P/name)==digest, 'CPU route source changed: '+name
route_ops={lib:set(data['available_ops']) for lib,data in route_catalog['libraries'].items()}
i=100
while i<len(rows):''';assert s.count(needle)==1;s=s.replace(needle,new);needle="row=rows[i];group=(row['library'],row['op'])\n";new=needle+''' if row['library'] in route_ops and row['op'] not in route_ops[row['library']]:
  ledger[str(i)]={'result':{'ordinal':i,'key':[row['library'],row['op'],row['case']],'status':'provider_registry_route_missing','reason':'CPU-lazy package and original classic fallback have no registered provider for this operator; nativeAPI absence not inferred','evidence':'CPU_TLA_ROUTE_CATALOG.json'}}
  put(ledger_path,ledger);publish('CPU_missing_registry_route_no_device', [i]);i+=1;continue
''';assert s.count(needle)==1;s=s.replace(needle,new);compile(s,str(f),'exec');f.write_text(s)
 put('CURRENT_PROCESS.json',{'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entry':'continue_ordered.py','foreground_tool_session':'see CPU_ROUTE_TOOL.json','own_migration_disabled':True})
put('CPU_ROUTE_HANDOFF.json',{'status':'resumed_with_proved_CPU_registry_preflight','previous_pid':pid,'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
