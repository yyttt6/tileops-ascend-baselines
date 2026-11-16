from pathlib import Path
import json,hashlib,shutil
from nn_layernorm_math_retention import capture as norm_capture
S=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def capture(out):
 out=Path(out);norm_capture(out);records={}
 for label,root,mname,private in [('Mul',S/'ops-math-dequant-v1','DEQUANT_PACKAGE.json','private-v1'),('Add',S/'ops-math-add-v1','OPSMATH_ADD_NATIVE_PACKAGE_V2.json','private-v2')]:
  m=json.loads((root/mname).read_text());folder=out/label;folder.mkdir();shutil.copy2(root/mname,folder/mname);items=m.get('all_files',m.get('files',{}));expected={}
  # Add manifest staging_root identifies the exact immutable deployed package.
  package=Path(m['staging_root']) if 'staging_root' in m else root/private
  for name,rec in items.items():
   src=package/name;digest=rec['sha256'] if isinstance(rec,dict) else rec;assert sha(src)==digest
   dst=folder/'package'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==digest;expected[name]=digest
  records[label]={'package_root':str(package),'manifest':mname,'manifest_sha256':sha(root/mname),'files':expected,'device_objects':m.get('device_objects',m.get('objects'))}
 (out/'MUL_ADD_RETENTION.json').write_text(json.dumps(records,indent=2))
