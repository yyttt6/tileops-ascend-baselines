from pathlib import Path
import json,sys,hashlib,subprocess,shlex,tarfile,datetime,re
D=Path('reports/baseline-over3-campaign-20261110/ordered100-20261112').resolve();A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
s=json.loads((A/'SERVER1_GEMM_PAD4_QUALIFICATION_V3.json').read_text());assert s['exit_code']==0 and s['guard']['performance_valid'] and s['guard']['phase']=='finished';tag=s['tag'];f=A/s['result'];v=json.loads(f.read_text());assert len(v['checks'])==4 and v['full_output_and_inputs_verified'] and v['original_1GiB_flush_selfcheck_passed']
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1';pkg=json.loads((A/'GEMM_PAD4_NATIVE_PACKAGE.server1.json').read_text());log=A/(tag+'.log');loads=[];groups=[];active=False;families=('MatMulV3_','BatchMatMulV3_','FusedMatMul_')
for index,line in enumerate(log.read_text().splitlines()):
 if line.startswith('BASELINE_PROBE_CALL_BEGIN '):groups.append({'marker':json.loads(line.split(' ',1)[1]),'launches':[]});active=True
 if 'Register Kernel succ' in line and 'bin path:' in line:
  match=re.search(r'bin path: (.*\.o)',line)
  if match and Path(match[1]).name.startswith(families):loads.append((index,match[1],line))
 if active and 'StreamLaunchKernelV2:' in line:
  match=re.search(r'kernel_name=([^,]+)',line)
  if match and match[1].startswith(families):
   found=[v for v in loads if v[0]<index and match[1].startswith(Path(v[1]).stem+'_')];assert found,match[1];_,path,load=found[-1];assert path.startswith(pkg['staging_root']+'/op_impl/ai_core/tbe/kernel/'),path
   relative=str(Path(path).relative_to(pkg['staging_root']));h=pkg['all_files'][relative]['sha256'];actual=c.remote('sha256sum '+shlex.quote(path),stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==actual;groups[-1]['launches'].append({'kernel':match[1],'path':path,'sha256':h,'load_log':load,'launch_log':line})
 if line.startswith('BASELINE_PROBE_CALL_END '):active=False
assert len(groups)==4 and all(g['launches'] for g in groups),'Require actualprivate loadANDlaunch attribution per fullAPI probe'
lines=groups

hostname=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote('import os;print(os.uname().nodename)'),stdout=subprocess.PIPE,text=True,check=True).stdout.strip();proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':c.H,'execution_hostname':hostname,'private_original_device_kernels_verified':True,'not_formal_checker_or15raw':True,'package_manifest_sha256':hashlib.sha256((A/'GEMM_PAD4_NATIVE_PACKAGE.server1.json').read_bytes()).hexdigest(),'guard_sha256':hashlib.sha256((A/(tag+'-guard.json')).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'api_sha256':hashlib.sha256((D/'ops_nn_gemm_pad4_api.py').read_bytes()).hexdigest(),'private_kernel_registration_lines':lines,'scope':[490,494,496,497],'four_full_output_and_inputbit_checks':v['checks'],'runtime':v['runtime'],'other_host_functional_acceptance_not_assumed':True}
f=A/'GEMM_PAD4_H1_HOST_QUALIFICATION.json';f.write_text(json.dumps(proof,indent=2)+'\n');raw=f.read_bytes();dest=N+'/GEMM_PAD4_HOST_QUALIFICATION.json';code='from pathlib import Path;import sys;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)';c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=raw,check=True)
archive=A/(tag+'.qualification.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in [log,A/(tag+'-guard.json'),A/s['result'],A/(s['result']+'.maps.txt'),f,A/'GEMM_PAD4_PROBE_DEPLOYMENT.json',A/'probe_gemm_pad4_v3.py',D/'ops_nn_gemm_pad4_api.py',A/'gemm-pad4-probe-entry-v3.sh',A/'GEMM_PAD4_NATIVE_PACKAGE.server1.json']:t.add(p,arcname=p.name)
raw=archive.read_bytes();h=hashlib.sha256(raw).hexdigest();dest=N+'/'+archive.name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(p.read_bytes()).hexdigest())';r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert r.stdout.decode().strip()==h;(A/'GEMM_PAD4_H1_QUALIFICATION_ARCHIVE.json').write_text(json.dumps({'local':str(archive),'remote':dest,'sha256':h,'bytes':len(raw)},indent=2)+'\n');print('H1 Pad4 completeoriginalAPI+flush/privatekernelqualified',h)
