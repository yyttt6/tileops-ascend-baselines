from pathlib import Path
import json,subprocess,shlex,hashlib,re,datetime,tarfile
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1';s=json.loads((A/'SERVER1_MATH_L2_QUALIFICATION.json').read_text());assert s['exit_code']==0 and s['guard']['performance_valid'] is True and s['guard']['phase']=='finished';tag=s['tag'];result=A/s['result'];v=json.loads(result.read_text());assert v['full_output_and_inputs_verified'] and v['original_1GiB_flush_selfcheck_passed'] and len(v['checks'])==4
log=A/(tag+'.log');loads=[];launches=[];groups=[]
for idx,line in enumerate(log.read_text().splitlines()):
 if 'Register Kernel succ' in line and 'bin path:' in line:
  m=re.search(r'bin path: (.*\.o)',line)
  if m:loads.append((idx,m[1],line))
 if 'StreamLaunchKernelV2:' in line:
  m=re.search(r'kernel_name=([^,]+)',line)
  if not m or not m[1].startswith(('Mul_','ReduceSum_','Sqrt_')):continue
  found=[x for x in loads if x[0]<idx and m[1].startswith(Path(x[1]).stem+'_')];assert found,m[1];_,path,load=found[-1];kind=next(x for x in ('Mul','ReduceSum','Sqrt') if m[1].startswith(x+'_'));launches.append({'kind':kind,'kernel':m[1],'path':path,'load_line':load,'launch_line':line})
  if kind=='Sqrt':
   group=launches[-3:];assert [x['kind'] for x in group]==['Mul','ReduceSum','Sqrt'];assert all('/server2-aligned-stack/native/' in x['path'] and '/private' in x['path'] for x in group);groups.append(group)
assert len(groups)==4,len(groups)
paths=sorted({r['path'] for g in groups for r in g});code='import json,hashlib,os;from pathlib import Path;ps='+repr(paths)+';print(json.dumps({"hostname":os.uname().nodename,"files":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ps}}))';raw=subprocess.check_output(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)]);actual=json.loads(raw)
# All three original package manifests, including the reused Mul and ReduceSum, remain byte verified.
code='''from pathlib import Path
import json,hashlib
base=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');ms=[base/'ops-math-l2-sqrt-v1/MATH_L2_SQRT_PACKAGE.json',base/'ops-math-dequant-v1/DEQUANT_PACKAGE.json'];parts=json.loads((base/'ops-math-l1-v1/MATH_L1_PACKAGES.json').read_text());ms.append(Path(parts['Sum']['manifest']));out={}
for f in ms:
 m=json.loads(f.read_text());assert m['pin']=='64322f6487162c0934ec0364af57ed2f51175a03'
 for k,r in m['files'].items():
  p=Path(m['package'])/k;assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
 out[str(f)]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'package':m['package'],'files':m['files']}
print(json.dumps(out))
''';packages=json.loads(subprocess.check_output(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)]))
known={str(Path(m['package'])/k):r['sha256'] for m in packages.values() for k,r in m['files'].items()}
for p,h in actual['files'].items():assert known[p]==h,(p,h)
for g in groups:
 for r in g:r['sha256']=actual['files'][r['path']]
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_API_and_original_flush_and_private_kernel_verified':True,'execution_hostname':actual['hostname'],'package_manifest_sha256':hashlib.sha256((A/'MATH_L2_SQRT_PACKAGE.json').read_bytes()).hexdigest(),'four_complete_API_checks':v['checks'],'private_load_and_launch_per_group':groups,'three_original_package_manifests':{k:v['sha256'] for k,v in packages.items()},'guard_sha256':hashlib.sha256((A/(tag+'-guard.json')).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'not_formal_or15raw_or_coverage':True,'scope':[373,374,375,379]};p=A/'MATH_L2_H1_HOST_QUALIFICATION.json';p.write_text(json.dumps(proof,indent=2)+'\n');raw=p.read_bytes();code='from pathlib import Path;import sys;p=Path('+repr(N+'/MATH_L2_HOST_QUALIFICATION.json')+');b=sys.stdin.buffer.read();assert not p.exists();p.write_bytes(b)';subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],input=raw,check=True)
archive=A/(tag+'.qualification.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in [log,A/(tag+'-guard.json'),result,A/(s['result']+'.maps.txt'),A/'MATH_L2_H1_HOST_QUALIFICATION.json',A/'MATH_L2_SQRT_PACKAGE.json',A/'probe_math_l2.py',A/'math-l2-probe-entry.sh',D/'ops_math_l2_api.py',D/'ops_math_sqrt_api.py',D/'ops_math_l1_api.py',D/'ops_math_dequant_api.py']:t.add(p,arcname=p.name)
raw=archive.read_bytes();h=hashlib.sha256(raw).hexdigest();dest=N+'/'+archive.name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists();p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';r=subprocess.check_output(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],input=raw);assert r.decode().strip()==h;(A/'MATH_L2_H1_QUALIFICATION_ARCHIVE.json').write_text(json.dumps({'local':str(archive),'remote':dest,'sha256':h},indent=2)+'\n');print('Four full3API+inputbits+originalflush, private load+launch, dualarchive SHA verified')
