"""Move only explicit inactive campaign batches' accepted inputs; retain local gzip and remote full PT."""
from pathlib import Path
import json,subprocess,shlex,hashlib,gzip,fcntl
P=Path(__file__).resolve().parent;H='openlibing-DevEnv-661495'
lock=(P/'current-input-migration.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=P/'CURRENT_INPUT_STORAGE_MIGRATION.json'
done=json.loads(receipt.read_text()) if receipt.exists() else []
def sha(f):
 h=hashlib.sha256()
 while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
batches=['sgl-dequant-block-server1','sgl-dequant-channel-server1','sgl-dequant-current-server1','sgl-dequant-layout-server1','sgl-lerp-api-server1','sgl-neg-pad-server1','sgl-silu-pad-server1','vllm-rms-api-server1','vllm-rms-api-final-server1','vllm-rms-expand-server1','vllm-gemm-fp16-server1','vllm-fused-server1','vllm-neg-server1','vllm-cnn-server1','vllm-fused5120-server1','vllm-ffn-server1','vllm-silu-server1','vllm-sigmoid-server1','vllm-mean-current-server1','vllm-groupnorm-current-server1','vllm-adaln-server1','vllm-sum-current-server1','vllm-bmm-api-server1','vllm-avgpool-regular-server1','vllm-avgpool-overlap-server1','vllm-avgpool-ceil-server1','vllm-fusedqk-api-server1','vllm-ssd-coupling-v2-server1','sgl-rms-current-server1','sgl-sigmoid-current-server1','sgl-logical-cnn-server1','sgl-logical-float-cnn-server1','sgl-logical-remaining-server1','sgl-any-current-server1','sgl-all-current-server1','sgl-numeric-reduce-server1','sgl-argmax-current-server1','sgl-argmin-current-server1','sgl-amax-current-server1','sgl-amin-current-server1']
for name in batches:
 d=P/name;rows=json.loads((d/'RESULTS.json').read_text())
 for i in sorted({r['index'] for r in rows if r['status']=='formal_raw_verified'},reverse=True):
  key=[name,i]
  if any(x['key']==key for x in done):continue
  formal=next(r for r in rows if r['index']==i and r['status']=='formal_raw_verified')
  with (d/'remote/runs'/str(i)/formal['variant']/formal['attempt']/'result.json').open('rb') as f:assert sha(f)==formal['result_sha256']
  expected=json.loads((d/'remote/runs'/str(i)/'prepare.json').read_text())['payload_sha256']
  root='/opt/baseline-over3-20261110/'+name;original=f'{root}/runs/{i}/payload.pt';dest=f'/workspace/user_data/baseline-over3-20261110/completed-inputs/{name}/{i}/payload.pt'
  code="""import pathlib,hashlib,shutil,json
p=pathlib.Path(%r);d=pathlib.Path(%r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
assert sha(p)==%r
d.parent.mkdir(parents=True,exist_ok=True)
if not d.exists():shutil.copyfile(p,d)
assert sha(d)==%r
print(json.dumps({'original':str(p),'target':str(d),'bytes':d.stat().st_size}))
"""%(original,dest,expected,expected)
  info=json.loads(subprocess.check_output(['ssh',H,'python3 -c '+shlex.quote(code)],text=True))
  local=P/'completed-inputs'/f'{name}-{i}.pt.gz';local.parent.mkdir(exist_ok=True)
  if not local.exists():
   temp=local.with_suffix('.gz.partial')
   with temp.open('wb') as f:subprocess.run(['ssh',H,'gzip -c '+shlex.quote(dest)],stdout=f,check=True)
   with gzip.open(temp,'rb') as f:assert sha(f)==expected
   temp.replace(local)
  with gzip.open(local,'rb') as f:assert sha(f)==expected
  code="""import pathlib,hashlib,os
p=pathlib.Path(%r);d=pathlib.Path(%r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
assert sha(p)==%r and sha(d)==%r
if p.is_symlink():assert p.resolve()==d.resolve()
else:
 t=p.with_name('payload.pt.current-migration-link');t.symlink_to(d);os.replace(t,p)
"""%(original,dest,expected,expected)
  subprocess.run(['ssh',H,'python3 -c '+shlex.quote(code)],check=True)
  with local.open('rb') as f:compressed_sha=sha(f)
  done.append(dict(info,key=key,local_gzip=str(local),uncompressed_sha256=expected,local_gzip_sha256=compressed_sha))
  tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(done,indent=2)+'\n');tmp.replace(receipt)
  print('migrated',name,i,info['bytes'],flush=True)
