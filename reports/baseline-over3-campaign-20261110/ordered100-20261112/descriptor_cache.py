"""Trusted task-local lazy Case cache; invalidated by complete source-tree digest.
Only avoids CPU manifest reconstruction. Inputs, oracle and measured API stay live.
"""
import pathlib,hashlib,json,pickle,sys,platform

def fingerprint(root):
 h=hashlib.sha256();count=0
 for f in sorted(pathlib.Path(root).rglob('*')):
  if not f.is_file() or f.suffix not in ('.py','.yaml','.yml','.json','.toml'):continue
  if '__pycache__' in f.parts:continue
  h.update(str(f.relative_to(root)).encode()+b'\0');h.update(hashlib.sha256(f.read_bytes()).digest());count+=1
 return {'source_sha256':h.hexdigest(),'source_files':count,'python':sys.version,'machine':platform.machine()}

def write(case,path,root,torch_version,op):
 path=pathlib.Path(path);assert set(case.__dict__)=={'id','_call','_entry'},'cache descriptor before inputs/reference access'
 blob=pickle.dumps(case,protocol=5);identity=fingerprint(root);identity.update(torch_version=torch_version,key=[op,case.id],sha256=hashlib.sha256(blob).hexdigest())
 path.write_bytes(blob);path.with_suffix('.identity.json').write_text(json.dumps(identity,indent=2)+'\n')
 return identity

def load(path,root,torch_version,op,case_id):
 path=pathlib.Path(path);identity=json.loads(path.with_suffix('.identity.json').read_text());current=fingerprint(root)
 assert all(identity[k]==v for k,v in current.items()),'descriptor source/toolchain changed'
 assert identity['torch_version']==torch_version and identity['key']==[op,case_id]
 blob=path.read_bytes();assert hashlib.sha256(blob).hexdigest()==identity['sha256']
 case=pickle.loads(blob);assert case.id==case_id and set(case.__dict__)=={'id','_call','_entry'}
 return case
