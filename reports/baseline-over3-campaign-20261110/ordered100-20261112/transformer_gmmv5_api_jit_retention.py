from pathlib import Path
import json,hashlib,shutil,datetime
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def capture(work,destination):
 work=Path(work);out=Path(destination);out.mkdir(parents=True,exist_ok=False);records=[]
 for meta in sorted(work.rglob('*.json')):
  data=json.loads(meta.read_text())
  if not isinstance(data,dict) or 'binFileName' not in data or 'sha256' not in data:continue
  name=data['binFileName'];assert Path(name).name==name;obj=meta.with_name(name+data.get('binFileSuffix','.o'))
  if not obj.is_file():continue
  files={}
  for p in meta.parent.glob(name+'*'):
   if p.is_file():
    dst=out/p.name
    if dst.exists():assert dst.read_bytes()==p.read_bytes()
    else:shutil.copy2(p,dst)
    files[p.name]={'sha256':sha(dst),'bytes':dst.stat().st_size}
  assert sha(obj)==data['sha256'];records.append({'kernel':name,'metadata':data,'device_ELF_sha256':sha(obj),'retained_files':files})
 assert records,'No original V5 generated kernel artifacts captured; retain raw and diagnose, no fake qualification'
 lib=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-original-v1/private-v1/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so');maps=Path('/proc/self/maps').read_text();assert str(lib) in maps;shutil.copy2(lib,out/'loaded-libcust_opapi.so');(out/'loaded-maps.txt').write_text(maps)
 r={'loaded_library_path':str(lib),'loaded_library_sha256':sha(lib),'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outside_timed_windows':True,'records':records,'source_or_flags_or_tensor_changes':False};(out/'RETENTION.json').write_text(json.dumps(r,indent=2)+chr(10));return r
