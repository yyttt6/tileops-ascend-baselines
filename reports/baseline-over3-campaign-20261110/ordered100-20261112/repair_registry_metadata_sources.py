"""Bounded exact small-source restoration; never changes existing files or native binaries."""
import pathlib,json,hashlib,subprocess,shlex,datetime,sys
P=pathlib.Path(__file__).resolve().parent;ROOT=P.parents[2];R='/opt/baseline-over3-20261110/ordered100-20261112';H='openlibing-DevEnv-661495';PREFIX='/root/dyq/workspace/reports/baseline-integration-700-20261026/'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def put(x):
 f=P/'REGISTRY_METADATA_SOURCE_REPAIR.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(f)
state={'started_at':stamp(),'metadata_only':True,'device_imports_extensions_and_child_execution_denied_by_probe':True,'native_or_kernel_files_copied':False,'existing_remote_files_overwritten':False,'accepted_cases_incremented':0,'actions':[],'libraries':{}};put(state)
for lib in ['ops_nn','ops_transformer','ops_math']:
 for step in range(24):
  cmd='/opt/tileops-ci/runtime/bin/python -I -S '+R+'/registry_metadata_preflight.py '+lib
  proc=subprocess.run(['ssh','-o','ConnectTimeout=10',H,cmd],text=True,capture_output=True,timeout=45)
  log=P/f'registry-metadata-{lib}-{step}.CPU.log';log.write_text(proc.stdout+proc.stderr)
  assert proc.returncode==0,'CPU boundary/probe failed; no blind device fallback'
  result=json.loads(proc.stdout.splitlines()[-1]);state['libraries'][lib]=result;state['checked_at']=stamp();put(state)
  print(lib,step,result['status'],result.get('missing_file'),flush=True)
  if result['status']=='registry_import_passed':break
  missing=result.get('missing_file')
  if not missing or not missing.startswith(PREFIX):break
  path=pathlib.Path(missing)
  if '..' in path.parts or path.suffix not in ('.json','.py') or not path.is_file():break
  data=path.read_bytes()
  if len(data)>2_000_000:break
  rel=path.relative_to(pathlib.Path(PREFIX));dest=R+'/registry-dependencies-v1/'+str(rel);digest=sha(data)
  code='''import pathlib,sys,hashlib,json
original=pathlib.Path(sys.argv[1]);private=pathlib.Path(sys.argv[2]);expected=sys.argv[3];data=sys.stdin.buffer.read()
assert hashlib.sha256(data).hexdigest()==expected
prefix=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026')
assert original.is_relative_to(prefix)
assert original.suffix in ('.py','.json') and len(data)<=2_000_000
if original.exists() or original.is_symlink():
 assert original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest()==expected,'existing original is never overwritten'
else:
 private.parent.mkdir(parents=True,exist_ok=True)
 if private.exists():assert hashlib.sha256(private.read_bytes()).hexdigest()==expected
 else:
  with private.open('xb') as out:out.write(data)
 original.parent.mkdir(parents=True,exist_ok=True)
 assert original.parent.resolve().is_relative_to(prefix.resolve())
 original.symlink_to(private)
assert hashlib.sha256(original.read_bytes()).hexdigest()==expected
print(json.dumps({'original':str(original),'private':str(private),'sha256':expected,'bytes':len(data),'resolved':str(original.resolve())}))
'''
  command='python3 -c '+shlex.quote(code)+' '+shlex.quote(missing)+' '+shlex.quote(dest)+' '+shlex.quote(digest)
  restored=subprocess.run(['ssh','-o','ConnectTimeout=10',H,command],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30);assert restored.returncode==0,restored.stderr.decode()
  record=json.loads(restored.stdout);record.update({'library':lib,'checked_at':stamp(),'local_source':str(path),'local_resolved':str(path.resolve()),'scope':'small pinned registry/Python metadata dependency, not performance or native acceptance'})
  state['actions'].append(record);put(state)
 else:state['libraries'][lib]['bounded_restore_limit_reached']=True
state['finished_at']=stamp();state['status']='CPU_registry_source_restoration_reviewed_native_qualification_still_separate';put(state);print('FINISHED',len(state['actions']),'smallfiles, no performance acceptance',flush=True)
