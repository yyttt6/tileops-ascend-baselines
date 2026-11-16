from pathlib import Path
import subprocess,json,hashlib,gzip,shlex
import sys
SUB=sys.argv[1];assert SUB=='sgl-dispatch-smoke-server1'
P=Path(__file__).resolve().parent;D=P/SUB;H='openlibing-DevEnv-661495';R='/opt/baseline-over3-20261110/'+SUB;S='/workspace/user_data/baseline-over3-20261110/completed-inputs/'+SUB
rs=json.loads((D/'RESULTS.json').read_text());receipts=[]
for i in [225,220,219,218,155]:
 for v in ['sgl_fixed']:assert any(r['index']==i and r['variant']==v and r['status']=='pilot_raw_verified' for r in rs)
 prep=json.loads((D/'remote/runs'/str(i)/'prepare.json').read_text());expected=prep['payload_sha256'];dest=f'{S}/{i}/payload.pt';original=f'{R}/runs/{i}/payload.pt'
 cmd=f'mkdir -p {S}/{i} && cp -n {original} {dest} && sha256sum {original} {dest}'
 hashes=subprocess.check_output(['ssh',H,cmd],text=True).splitlines();assert all(x.split()[0]==expected for x in hashes)
 local=P/'completed-inputs'/f'{SUB}-{i}.pt.gz';local.parent.mkdir(exist_ok=True)
 with local.open('wb') as f:subprocess.run(['ssh',H,'gzip -c '+dest],stdout=f,check=True)
 h=hashlib.sha256()
 with gzip.open(local,'rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 assert h.hexdigest()==expected
 code="import pathlib,hashlib,os; p=pathlib.Path(%r); d=pathlib.Path(%r); assert hashlib.sha256(p.read_bytes()).hexdigest()==%r; assert hashlib.sha256(d.read_bytes()).hexdigest()==%r; t=p.with_name('payload.pt.migration-link'); t.symlink_to(d); os.replace(t,p)"%(original,dest,expected,expected)
 subprocess.run(['ssh',H,'python3 -c '+shlex.quote(code)],check=True)
 receipts.append(dict(original=original,target=dest,local_gzip=str(local),uncompressed_sha256=expected,local_gzip_sha256=hashlib.sha256(local.read_bytes()).hexdigest()))
 (P/(SUB.upper()+'_PAYLOAD_STORAGE_MIGRATION.json')).write_text(json.dumps(receipts,indent=2)+'\n')
 print('verified and migrated',i,flush=True)
