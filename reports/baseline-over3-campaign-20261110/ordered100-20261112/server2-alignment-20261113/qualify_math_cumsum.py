from pathlib import Path
import json,re,hashlib,subprocess,shlex,datetime,tarfile
D=Path(__file__).resolve().parent.parent;A=D/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-cumscan-v1';V=N;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();s=json.loads((A/'SERVER1_MATH_CUMSUM_QUALIFICATION.json').read_text());assert s['exit_code']==0 and s['guard']['phase']=='finished' and s['guard']['performance_valid'] is True;r=json.loads((A/s['result']).read_text());assert len(r['checks'])==4 and r['full_output_and_inputs_verified'] and r['original_1GiB_flush_selfcheck_passed'];mf=A/'MATH_CUMSCAN_PACKAGE.json'
if not mf.exists():mf=A/'MATH_CUMSCAN_PACKAGE.server1.json'
# Package bytes are read from the exact already-qualified remote manifest, not reconstructed.
def remote(cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495',cmd],check=True,**kw)
raw=remote('cat '+V+'/MATH_CUMSCAN_PACKAGE.json',stdout=subprocess.PIPE).stdout;m=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==sha(A/'MATH_CUMSCAN_PACKAGE.json');root=m['package'];loads=[];groups=[];active=False
for idx,line in enumerate((A/(s['tag']+'.log')).read_text().splitlines()):
 if line.startswith('BASELINE_CUMSUM_CALL_BEGIN '):assert not active;active=True;groups.append({'marker':json.loads(line.split(' ',1)[1]),'launches':[]})
 if '[BinLoadImpl]' in line and 'Register Kernel succ' in line and 'bin path:' in line:
  hit=re.search(r'bin path: (.*\.o)',line)
  if hit and Path(hit[1]).name.startswith('Cumsum_'):loads.append((idx,hit[1],line))
 if active and 'StreamLaunchKernelV2:' in line:
  hit=re.search(r'kernel_name=([^,]+)',line)
  if hit and hit[1].startswith('Cumsum_'):
   matches=[x for x in loads if x[0]<idx and hit[1].startswith(Path(x[1]).stem+'_')];assert matches;rec=matches[-1];path=rec[1];assert path.startswith(root+'/');expected=m['files'][str(Path(path).relative_to(root))]['sha256'];actual=remote('sha256sum '+shlex.quote(path),stdout=subprocess.PIPE,text=True).stdout.split()[0];assert actual==expected;groups[-1]['launches'].append({'kernel':hit[1],'path':path,'sha256':actual,'load_line':rec[2],'launch_line':line})
 if line.startswith('BASELINE_CUMSUM_CALL_END '):assert active;active=False
assert len(groups)==4 and not active and all(g['launches'] for g in groups)
helpers=['ops_math_cumsum_api.py','ops_math_l1_api.py'];q={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_API_and_original_flush_and_private_kernel_verified':True,'execution_hostname':remote('hostname',stdout=subprocess.PIPE,text=True).stdout.strip(),'native_package_manifest_sha256':hashlib.sha256(raw).hexdigest(),'qualified_API_source_sha256':{name:sha(D/name) for name in helpers},'private_load_and_launch_per_complete_group':groups,'original_reference_and_full_inputs_verified':r['checks'],'guard_sha256':sha(A/(s['tag']+'-guard.json')),'result_sha256':sha(A/s['result']),'log_sha256':sha(A/(s['tag']+'.log')),'no_formal_15raw_or_baseline_coverage':True,'scope':'Cumsum256..258 complete original aclnnCumsum, original input/output dtype and axis, all contiguous/descriptor/nativeworkspace/output timed. No kernel/checker/tolerance changes.'}
for name,h in q['qualified_API_source_sha256'].items():assert remote('sha256sum '+str('/opt/baseline-over3-20261110/ordered100-20261112/'+name),stdout=subprocess.PIPE,text=True).stdout.split()[0]==h
qfile=A/'MATH_CUMSUM_H1_HOST_QUALIFICATION.json';assert not qfile.exists();qfile.write_text(json.dumps(q,indent=2)+'\n')
def put(path,raw):
 remote('python3 -c '+shlex.quote('from pathlib import Path;import sys;p=Path('+repr(path)+');assert not p.exists() or p.read_bytes()==sys.stdin.buffer.peek();p.write_bytes(sys.stdin.buffer.read())'),input=raw)
put(N+'/CUMSUM_HOST_QUALIFICATION.json',qfile.read_bytes());assert remote('sha256sum '+N+'/CUMSUM_HOST_QUALIFICATION.json',stdout=subprocess.PIPE,text=True).stdout.split()[0]==sha(qfile)
archive=A/(s['tag']+'.qualification.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for p in [qfile,A/(s['tag']+'-guard.json'),A/s['result'],A/(s['result']+'.maps.txt'),A/(s['tag']+'.log'),A/'probe_math_cumsum.py',A/'MATH_CUMSCAN_PACKAGE.json',*[D/x for x in helpers]]:tar.add(p,arcname=p.name)
put(N+'/'+archive.name,archive.read_bytes());assert remote('sha256sum '+N+'/'+archive.name,stdout=subprocess.PIPE,text=True).stdout.split()[0]==sha(archive);(A/'MATH_CUMSUM_H1_QUALIFICATION_ARCHIVE.json').write_text(json.dumps({'local':str(archive),'remote':N+'/'+archive.name,'sha256':sha(archive)},indent=2)+'\n');print('Qualified4 fullAPI groups, privateactual load/launch, originalflush and dualarchive SHA')
