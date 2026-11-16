"""Read actual successful guarded call provenance; never execute another device probe."""
from pathlib import Path
import json,re,hashlib,subprocess,shlex,datetime
D=Path(__file__).resolve().parent.parent;A=D/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-poolmean-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((A/'SERVER1_POOLMEAN_ATTRIBUTION.json').read_text());assert s['status']=='functional_probe_passed_not_formal_baseline' and s['guard']['performance_valid'] and s['exit_code']==0;r=json.loads((A/s['result']).read_text());assert len(r['rows'])==3 and all(x['outputs']==1 and x['status']=='full_outputs_and_inputs_verified' for x in r['rows']);assert r['API_binding_sha256']==sha(D/'ops_nn_poolmean_api.py');m=json.loads((A/'POOLMEAN_NATIVE_PACKAGE.server2.json').read_text());assert m['source_tracked_files_verified']==25610 and m['source_changes']==[] and m['declared_dependency']['scope']=='outputSize=[1,1] only'
lines=(A/(s['tag']+'.log')).read_text().splitlines();loads=[];launches=[];groups=[];active=None
for index,line in enumerate(lines):
 if line.startswith('BASELINE_PROBE_CALL_BEGIN '):active=json.loads(line.split(' ',1)[1]);groups.append({'marker':active,'launches':[]})
 if ('workspace num is' in line and 'bin path is' in line) or ('[BinLoadImpl]' in line and 'Register Kernel succ' in line and 'bin path:' in line):
  h=re.search(r'bin path[: ]+(?:is )?(.*\.o)',line)
  if h and Path(h[1]).name.startswith(('ReduceMean_','ReduceMeanD_')):loads.append((index,h[1],line))
 if active is not None and 'StreamLaunchKernelV2:' in line:
  h=re.search(r'kernel_name=([^,]+)',line)
  if h and h[1].startswith(('ReduceMean_','ReduceMeanD_')):
   matches=[p for p in loads if p[0]<index and h[1].startswith(Path(p[1]).stem+'_')];assert matches,('actual kernel load source unknown',h[1]);rec=matches[-1];proof={'kernel':h[1],'path':rec[1],'load_log':rec[2],'launch_log':line};groups[-1]['launches'].append(proof);launches.append(proof)
 if line.startswith('BASELINE_PROBE_CALL_END '):active=None
assert len(groups)==3 and all(g['launches'] for g in groups),'Every complete original API call requires actual declared ReduceMean launch evidence'
paths={p['path'] for p in launches};paths.update(p for p in r['loaded_library_sha256'] if Path(p).name.startswith('libopapi_math.so'));assert any(Path(p).name.startswith('libopapi_math.so') for p in paths),'Missing actual SDK dependency library identity'
verified={}
for p in sorted(paths):
 assert '/opp/' in p or Path(p).name.startswith('libopapi_math.so'),('unexpected dependency root requires review',p)
 digests={h:subprocess.check_output(['ssh',h,'sha256sum '+shlex.quote(p)],text=True).split()[0] for h in ['openlibing-DevEnv-661495','openlibing-DevEnv-055409']};assert len(set(digests.values()))==1,('DeclaredSDK dependency differs from canonicalH2',p,digests);verified[p]=next(iter(digests.values()))
q={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_original_api_declared_dependencies_verified':True,'private_original_device_kernels_verified':False,'native_input_dtype':'torch.float32','scope':'exact original AdaptiveAvgPool2d output1x1 API; declared locked SDK ReduceMean dependency. All12 timedwrapper contracts remain original, actual formal checker pending.','declared_SDK_dependency_paths_sha256':verified,'actual_call_load_launch_proof':groups,'source_branch':m['declared_dependency'],'runtime':r['runtime'],'guard':s['guard'],'package_manifest_sha256':sha(A/'POOLMEAN_NATIVE_PACKAGE.server2.json'),'API_binding_sha256':r['API_binding_sha256'],'functional_result_sha256':sha(A/s['result']),'functional_log_sha256':sha(A/(s['tag']+'.log')),'not_source_rebuilt_SDK_math_or_private_pool_kernel':True}
p=A/'POOLMEAN_DECLARED_DEPENDENCY_QUALIFICATION.json';assert not p.exists();p.write_text(json.dumps(q,indent=2)+'\n');dst=N+'/'+p.name;code='from pathlib import Path;import sys;p=Path('+repr(dst)+');assert not p.exists();p.write_bytes(sys.stdin.buffer.read())';subprocess.run(['ssh','openlibing-DevEnv-661495','python3 -c '+shlex.quote(code)],input=p.read_bytes(),check=True);assert subprocess.check_output(['ssh','openlibing-DevEnv-661495','sha256sum '+shlex.quote(dst)],text=True).split()[0]==sha(p)
print('Qualified complete original API and declared actual SDK dependency; not privatekernel claim or formal coverage')
