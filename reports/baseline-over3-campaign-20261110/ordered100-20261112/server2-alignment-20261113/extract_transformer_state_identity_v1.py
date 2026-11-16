from pathlib import Path
import json,hashlib,struct,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-state-original-v1');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=json.loads((N/'BUILD_RECIPE_RESULT.json').read_text());assert r['compile_exit_code']==r['link_exit_code']==0;obj=N/'original_transformer_state.o';lib=N/'liboriginal_state_passing.so';assert sha(obj)==r['object_sha256'] and sha(lib)==r['binary_sha256'];symbol='kernel_cust_chunk_state_passing'
def elfs(p):
 raw=p.read_bytes();out={};start=1
 while True:
  i=raw.find(b'\x7fELF',start)
  if i<0:break
  start=i+4
  if raw[i+4:i+6]!=b'\x02\x01':continue
  machine=struct.unpack_from('<H',raw,i+18)[0]
  if machine!=4137:continue
  shoff=struct.unpack_from('<Q',raw,i+40)[0];entsize,n=struct.unpack_from('<HH',raw,i+58);end=shoff+entsize*n
  for j in range(n):
   h=struct.unpack_from('<IIQQQQIIQQ',raw,i+shoff+j*entsize)
   if h[1]!=8:end=max(end,h[4]+h[5])
  data=raw[i:i+end];assert symbol.encode() in data;digest=hashlib.sha256(data).hexdigest();out[digest]={'offset':i,'bytes':end,'machine':machine,'kernel_symbol':symbol,'data':data}
 return out
a=elfs(obj);b=elfs(lib);assert a and a.keys()==b.keys();folder=N/'device-identity';assert not folder.exists();folder.mkdir();device={}
for k,(h,v) in enumerate(sorted(a.items())):
 p=folder/('original-device-'+str(k)+'.elf');p.write_bytes(v['data']);assert sha(p)==h;device[str(p)]={'sha256':h,'object':{k:x for k,x in v.items() if k!='data'},'linked_so':{k:x for k,x in b[h].items() if k!='data'}}
sources=json.loads((N/'ORIGINAL_SOURCE.json').read_text());assert all(sha(N/'source'/n)==h for n,h in sources['files'].items());out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_source_sha256':{'source/'+n:h for n,h in sources['files'].items()},'binary_sha256':sha(lib),'object_sha256':sha(obj),'build_result_sha256':sha(N/'BUILD_RECIPE_RESULT.json'),'device_ELFs':device,'kernel_symbol':symbol,'native_load_or_numeric_acceptance':False,'source_or_kernel_or_flags_edits':False};assert not (N/'FORMAL_BUILD_IDENTITY.json').exists();(N/'FORMAL_BUILD_IDENTITY.json').write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out))
