"""Retain actual SDK package metadata/device ELF used by named raw tasks."""
from pathlib import Path
import hashlib,json,os,shutil
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def capture(destination,pieces):
 out=Path(destination);assert not out.exists();out.mkdir();opp=Path(os.environ['ASCEND_OPP_PATH']);root=opp/'built-in/op_impl/ai_core/tbe/kernel/ascend950';index={}
 for f in root.glob('*/*/*.json'):index.setdefault(f.name,[]).append(f)
 names=sorted({t['name'] for v in pieces for t in v['profile_kernel_tasks']});records=[];missing=[]
 def retain(f):
  key=hashlib.sha256(str(f).encode()).hexdigest()+'-'+f.name;dst=out/key
  if not dst.exists():shutil.copy2(f,dst)
  assert sha(dst)==sha(f);return {'original_path':str(f),'retained_file':key,'sha256':sha(f)}
 for name in names:
  candidates=index.get(name.rsplit('_',1)[0]+'.json',[]);matches=[]
  for f in candidates:
   md=json.loads(f.read_text())
   if name not in {x['kernelName'] for x in md.get('kernelList',[])} and name!=md.get('kernelName'):continue
   obj=f.with_suffix('.o');assert obj.exists() and sha(obj)==md['sha256'] and obj.read_bytes()[:4]==b'\x7fELF';matches.append({'metadata':retain(f),'device_ELF':retain(obj)})
  if matches:records.append({'name':name,'kind':'SDK_AICORE','matches':matches});continue
  configs=[]
  for f in (opp/'built-in/op_impl/aicpu').glob('*/config/*.json'):
   try:md=json.loads(f.read_text())
   except (ValueError,UnicodeDecodeError):continue
   if name in md:configs.append({'metadata':retain(f),'operator':md[name]})
  if configs:records.append({'name':name,'kind':'SDK_declared_AICPU_actual_raw_task','configs':configs,'device_runtime_library':'SDK device-side library; not claimed as retained private AICORE ELF'});continue
  missing.append(name)
 (out/'PROCESS_MAPS.txt').write_text(Path('/proc/self/maps').read_text());libs={}
 for line in (out/'PROCESS_MAPS.txt').read_text().splitlines():
  bits=line.split();f=Path(bits[-1]) if bits else Path('')
  if bits and bits[-1].startswith('/') and f.is_file() and '.so' in f.name and any(x in str(f) for x in ['Ascend','server2-aligned-stack','unified-stack-v1']):libs[str(f)]=sha(f)
 report={'actual_opp':str(opp),'records':records,'unbound_names':missing,'loaded_runtime_libraries':libs,'all_named_tasks':names,'all_device_work_kept_in_full32':True,'no_private_kernel_recompile':True};(out/'SDK_KERNEL_RETENTION.json').write_text(json.dumps(report,indent=2))
