from pathlib import Path
import json,re,hashlib,subprocess,shlex,datetime
p=Path('reports/baseline-over3-campaign-20261110/ordered100-20261112');a=p/'server2-alignment-20261113';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();s=json.loads((a/'SERVER1_OPSMATH_ADD_MARKER_ATTRIBUTION.json').read_text());assert s['status']=='functional_probe_passed_not_formal_baseline' and s['exit_code']==0 and s['guard']['performance_valid'];m=json.loads((a/'OPSMATH_ADD_NATIVE_PACKAGE_V2.server2.json').read_text());old=json.loads((a/'OPSMATH_ADD_NATIVE_PACKAGE.server2.json').read_text());base=json.loads((a/'PRIVATE_OPSMATH_ADD_QUALIFICATION.json').read_text());assert base['private_original_device_kernels_verified'];root=m['staging_root'];r=json.loads((a/s['result']).read_text());assert len(r['rows'])==1 and r['rows'][0]['dtype']=='torch.int32';assert r['API_binding_sha256']==sha(p/'ops_math_add_api_v2.py')
for k,v in old['all_files'].items():
 if k.endswith('binary_info_config.json') or k.endswith('/add_apt.json'):continue
 assert m['all_files'][k]['sha256']==v['sha256'],k
lines=(a/(s['tag']+'.log')).read_text().splitlines();loads=[];launches=[]
for idx,line in enumerate(lines):
 if ('workspace num is' in line and 'bin path is' in line) or ('[BinLoadImpl]' in line and 'Register Kernel succ' in line and 'bin path:' in line):
  hit=re.search(r'bin path[: ]+(?:is )?(.*\.o)',line)
  if hit and Path(hit.group(1)).name.startswith('Add_INT32_'):loads.append((idx,hit.group(1),line))
 if 'StreamLaunchKernelV2:' in line:
  hit=re.search(r'kernel_name=([^,]+)',line)
  if hit and hit.group(1).startswith('Add_INT32_'):launches.append((idx,hit.group(1),line))
assert len(launches)==1,(len(loads),len(launches));idx,kernel,line=launches[0];matches=[x for x in loads if x[0]<idx and kernel.startswith(Path(x[1]).stem+'_')];assert matches;rec=matches[-1];path=rec[1];assert path.startswith(root+'/');rel=str(Path(path).relative_to(root));expected=m['all_files'][rel]['sha256'];actual=subprocess.check_output(['ssh','openlibing-DevEnv-661495','sha256sum '+shlex.quote(path)],text=True).split()[0];assert actual==expected
objects={};
for oldpath,digest in base['actual_private_device_paths_sha256'].items():
 rel=str(Path(oldpath).relative_to(old['staging_root']));assert m['all_files'][rel]['sha256']==digest;objects[root+'/'+rel]=digest
objects[path]=actual;assert len(objects)==4
q={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'private_original_device_kernels_verified':True,'scope':'Original complete floating Add4 plus unchanged formal timer INT32 Add marker dependency','package_manifest_sha256':sha(a/'OPSMATH_ADD_NATIVE_PACKAGE_V2.server2.json'),'API_binding_sha256':sha(p/'ops_math_add_api_v2.py'),'base_floating_qualification_sha256':sha(a/'PRIVATE_OPSMATH_ADD_QUALIFICATION.json'),'base_floating_outputs_and_inputs_reused_only_after_exact_binary_source_header_identity':True,'actual_private_device_paths_sha256':objects,'marker_guard':s['guard'],'marker_load_launch_proof':{'load':rec[2],'launch':line,'path':path,'sha256':actual},'marker_functional_result_sha256':sha(a/s['result']),'marker_log_sha256':sha(a/(s['tag']+'.log')),'no_formal15_or_speedup_acceptance':True};f=a/'PRIVATE_OPSMATH_ADD_QUALIFICATION_V2.json';assert not f.exists();f.write_text(json.dumps(q,indent=2)+'\n');n='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1';subprocess.run(['ssh','openlibing-DevEnv-661495','cat > '+n+'/'+f.name],input=f.read_bytes(),check=True);print('QUALIFIED_ADD_FLOATING3_AND_ORIGINAL_INT32_MARKER',len(objects),flush=True)
