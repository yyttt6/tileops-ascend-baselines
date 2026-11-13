"""Invoke with copied environment interpreter, after reviewed environment setup."""
import hashlib, importlib.metadata, json, pathlib, sys

def verify(expected, original, backend_sha):
 expected=pathlib.Path(expected).resolve();original=pathlib.Path(original).resolve()
 assert pathlib.Path(sys.prefix).resolve()==expected
 assert pathlib.Path(sys.executable).resolve().is_relative_to(expected)
 for entry in sys.path:
  if entry:assert not pathlib.Path(entry).resolve().is_relative_to(original),entry
 fingerprint=json.loads((expected.parent/(expected.name+'.fingerprint.json')).read_text())
 assert fingerprint['prefix']==str(expected) and fingerprint['backend_sha']==backend_sha
 for name,h in fingerprint['files'].items():
  file=(expected/name).resolve();assert file.is_relative_to(expected),name
  assert hashlib.sha256(file.read_bytes()).hexdigest()==h,name
 import torch,torch_npu
 import torch._inductor.fx_passes.post_grad as pg
 modules={}
 for module in [torch,torch_npu,pg]:
  path=pathlib.Path(module.__file__).resolve();assert path.is_relative_to(expected),str(path)
  modules[module.__name__]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 assert modules[pg.__name__]['sha256']==backend_sha
 deps=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])
 return {'executable':sys.executable,'prefix':sys.prefix,'sys_path':sys.path,'modules':modules,'dependencies':deps}
if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
