from pathlib import Path
import hashlib,json,datetime,shutil,inspect,copy
objects={};handles={};calls={};actual_keys={};current=None;installed=False
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def begin(index):
 global current
 current=int(index);calls.setdefault(current,[])
def end():
 global current
 current=None
def install():
 global installed
 if installed:return
 import triton.compiler.compiler as c
 C=c.CompiledKernel;assert C.launch_enter_hook is None and C.launch_exit_hook is None
 old_init=C.__init__;old_handles=C._init_handles;original_key=c.get_cache_key
 def get_key(src,backend,options,env_vars):
  key=original_key(src,backend,options,env_vars=env_vars);digest=hashlib.sha256(key.encode('utf-8')).hexdigest();actual_keys[digest]={'text':key,'source_hash':src.hash(),'options':copy.deepcopy(options.__dict__),'environment':copy.deepcopy(env_vars)};return key
 c.get_cache_key=get_key
 def init(self,src,group,key):
  old_init(self,src,group,key);fn=getattr(getattr(src,'fn',None),'fn',None);file=Path(inspect.getsourcefile(fn));assert str(file).endswith('/fla/ops/triton/triton_core/causal_conv1d.py'),str(file)
  meta=self.metadata._asdict();backend=c.make_backend(self.metadata.target);actual=actual_keys[key];assert actual['source_hash']==src.hash();options=backend.parse_options(actual['options']);env=actual['environment'];rebuilt=hashlib.sha256(original_key(src,backend,options,env_vars=env).encode('utf-8')).hexdigest();assert rebuilt==key==meta['hash'],('actual original options Triton cache key mismatch',rebuilt,key)
  paths={str(k):str(v) for k,v in group.items()};assert any(Path(v).is_file() and Path(v).read_bytes()==self.kernel for v in paths.values());record={'hash':key,'recomputed_hash':rebuilt,'source_file':str(file),'source_sha256':sha(file),'function_name':self.name,'kernel_name':meta['kernel_name'],'binary_sha256':hashlib.sha256(self.kernel).hexdigest(),'metadata_files':paths,'compiler_file':c.__file__,'compiler_sha256':sha(c.__file__),'unchanged_options':dict(options.__dict__),'actual_original_cache_key_text':actual['text'],'post_compile_metadata_options_delta':{k:[v,meta.get(k)] for k,v in actual['options'].items() if v!=meta.get(k)},'invalidating_environment':env,'source_hash':src.hash()};objects[id(self)]=record
 def init_handles(self):
  old_handles(self)
  if id(self) in objects:handles[str(self.function)]=objects[id(self)]['hash']
 def launch(meta):
  data=meta if isinstance(meta,dict) else meta.get();key=handles[str(data['function'])]
  if current is not None:calls[current].append({'hash':key,'name':data['name'],'function':str(data['function'])})
 C.__init__=init;C._init_handles=init_handles;C.launch_enter_hook=launch;installed=True
 def stop():pass
 return True
def capture(work,destination):
 out=Path(destination);out.mkdir(parents=True,exist_ok=False);records={};files={}
 def keep(src,rel):
  p=Path(src);dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);files[str(rel)]={'sha256':sha(dst),'bytes':dst.stat().st_size};return str(rel)
 for r in objects.values():
  if r['hash'] in records:continue
  item=dict(r);item['retained_files']={};item['source_copy']=keep(r['source_file'],Path('sources')/(r['source_sha256']+'.py'));item['compiler_copy']=keep(r['compiler_file'],Path('compiler')/(r['compiler_sha256']+'.py'))
  for name,p in r['metadata_files'].items():item['retained_files'][name]=keep(p,Path('kernels')/r['hash']/Path(p).name)
  assert any(files[rel]['sha256']==r['binary_sha256'] for rel in item['retained_files'].values());records[r['hash']]=item
 assert records,'No actual original Triton kernel compiler identity captured';formal={str(i):calls.get(i,[]) for i in range(4,19)};assert all(formal.values()),'All15 original measured calls must have actual target launches'
 receipt={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outside_timed_windows':True,'actual_source_cache_key_loaded_binary_launches_verified':True,'records':records,'calls':{str(k):v for k,v in calls.items()},'formal_calls':formal,'files':files,'no_native_call_or_flags_or_math_changes':True};(out/'RETENTION.json').write_text(json.dumps(receipt,indent=2,default=str)+chr(10));return receipt
