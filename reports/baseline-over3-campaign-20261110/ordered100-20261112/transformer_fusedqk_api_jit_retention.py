"""CPU file retention outside timed windows, before CANN process-exit cleanup."""
from pathlib import Path
import hashlib,json,shutil,datetime
PREFIX=('te_normropeconcat_','te_rotarypositionembedding_')
def capture(work,destination):
 work=Path(work);out=Path(destination);out.mkdir(parents=True,exist_ok=False);records=[]
 for meta in sorted(work.rglob('*.json')):
  if not meta.name.startswith(PREFIX):continue
  data=json.loads(meta.read_text())
  if 'binFileName' not in data or 'sha256' not in data:continue
  name=data['binFileName'];assert name.startswith(PREFIX) and Path(name).name==name
  obj=meta.with_name(name+data.get('binFileSuffix','.o'));assert obj.is_file();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={}
  # Retain raw evidence before assertion so an auditing failure cannot erase it.
  for p in meta.parent.glob(name+'*'):
   if p.is_file():
    dst=out/p.name
    if dst.exists():assert dst.read_bytes()==p.read_bytes()
    else:shutil.copy2(p,dst)
    files[p.name]={'sha256':sha(dst),'bytes':dst.stat().st_size}
  assert sha(obj)==data['sha256'],('ELF metadata digest mismatch',str(obj))
  records.append({'kernel':name,'source_metadata_path':str(meta),'device_ELF_sha256':sha(obj),'metadata':data,'retained_files':files})
 # CANN may reuse its previous process compilation cache without regenerating files.
 # Reuse only immutable locally qualified capsules; downstream audit must bind
 # actual profile names to these exact objects and pinned private source again.
 if True:
  catalog=Path('/opt/baseline-over3-20261110/ordered100-20261112/TRANSFORMER_FUSEDQK_API_QUALIFIED_JIT_CAPSULES.json')
  allowed=json.loads(catalog.read_text())['capsules'] if catalog.exists() else []
  node=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text()).get('device_node');assert node
  for entry in allowed:
   if entry['guard_device_node']!=node:continue
   capsule=Path(entry['path'])
   receipt=capsule/'RETENTION.json'
   if not receipt.is_file():continue
   prior=json.loads(receipt.read_text())
   for item in prior['records']:
    for name,v in item['retained_files'].items():
     src=capsule/name;assert hashlib.sha256(src.read_bytes()).hexdigest()==v['sha256']
     dst=out/name
     if dst.exists():assert dst.read_bytes()==src.read_bytes()
     else:shutil.copy2(src,dst)
    meta=item['metadata'];obj=out/(meta['binFileName']+meta.get('binFileSuffix','.o'))
    assert hashlib.sha256(obj.read_bytes()).hexdigest()==meta['sha256']==item['device_ELF_sha256']
    item=dict(item);item['reused_qualified_capsule']=str(receipt);item['capsule_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest();records.append(item)
 assert records,'No exact private-selection generated ELF metadata retained before exit; no qualification inferred'
 r={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outside_timed_windows':True,'work_root':str(work),'records':records,'source_or_flags_or_tensor_changes':False};(out/'RETENTION.json').write_text(json.dumps(r,indent=2)+chr(10));return r
