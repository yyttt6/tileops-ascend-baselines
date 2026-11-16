"""CPU-only original CMake artifact packaging. Does not import device libraries."""
from pathlib import Path
import subprocess,shlex,json,hashlib,datetime
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-mean-v1'
g=json.loads((A/'MATH_POOL_MEAN_COMPILE_AFTER_HOST.json').read_text());assert g['status']=='original4_FP32_commands_compiled_pending_package_fullAPIqualification' and g['last_exit_code']==0 and g['last_guard']['performance_valid']
code=r'''from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
n=Path(NPATH);b=n/'build-host-v1';root=n/'private-v1';v=root/'vendors/allpairs_math_poolmean_math';assert not root.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source=json.loads((n/'SOURCE_MANIFEST.json').read_text())
for k,r in source['files'].items():assert sha(n/'source'/k)==r['sha256'],k
m=json.loads((n/'MEAN_COMPILE_SELECTION.json').read_text());assert len(m['actual_objects'])==4 and all(r['data_input_dtypes']==['float32'] for r in m['selected_whole_original_commands'])
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
copy(b/'libcust_opapi.so',v/'op_api/lib/libcust_opapi.so');copy(n/'source/math/reduce_mean/op_api/aclnn_mean.h',v/'op_api/include/aclnn_mean.h');copy(b/'libcust_opsproto_rt2.0.so',v/'op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so')
t=v/'op_impl/ai_core/tbe';copy(b/'libcust_opmaster_rt2.0.so',t/'op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so');(t/'op_tiling/liboptiling.so').symlink_to('lib/linux/aarch64/libcust_opmaster_rt2.0.so')
for part in ['dynamic','ascendc']:shutil.copytree(b/'tbe'/part,t/'allpairs_math_poolmean_math_impl'/part,symlinks=True)
info=list((b/'tbe').rglob('aic-ascend950-ops-info.json'));assert len(info)==1,info;copy(info[0],t/'config/ascend950/aic-ascend950-ops-info.json');k=b/'binary/ascend950/bin';shutil.copytree(k/'ascend950',t/'kernel/ascend950');shutil.copytree(k/'config',t/'kernel/config')
config=json.loads((t/'kernel/config/ascend950/binary_info_config.json').read_text());assert set(config)=={'ReduceMean'}
for r in config['ReduceMean']['binaryList']:
 for field in ['binPath','jsonPath']:assert (t/'kernel'/r[field]).is_file(),r[field]
exports=subprocess.check_output(['nm','-D','--defined-only',str(v/'op_api/lib/libcust_opapi.so')],text=True);assert all(x in exports for x in ['aclnnMeanGetWorkspaceSize','aclnnMean'])
objects={str(p.relative_to(root)):sha(p) for p in root.rglob('*.o')};assert len(objects)==4 and set(objects.values())==set(m['actual_objects'].values())
proof={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pin':source['pin'],'vendor':str(v),'package':str(root),'files':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in root.rglob('*') if p.is_file()},'objects':objects,'API_exports':['aclnnMeanGetWorkspaceSize','aclnnMean'],'source_manifest_sha256':sha(n/'SOURCE_MANIFEST.json'),'compile_selection_sha256':sha(n/'MEAN_COMPILE_SELECTION.json'),'original_source_unchanged':True,'runtime_qualified':False,'not_coverage_or_performance_acceptance':True,'H2newpackage_identity_not_claimed':True};(n/'MATH_POOL_MEAN_PACKAGE.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'files':len(proof['files']),'objects':len(objects),'manifest_sha256':sha(n/'MATH_POOL_MEAN_PACKAGE.json')}))
'''.replace('NPATH',repr(N))
r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8','openlibing-DevEnv-661495','/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],capture_output=True,text=True);(A/'MATH_POOL_MEAN_ASSEMBLE_CPU.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
for name in ['MATH_POOL_MEAN_PACKAGE.json','MEAN_COMPILE_SELECTION.json']:
 raw=subprocess.check_output(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8','openlibing-DevEnv-661495','cat '+N+'/'+name]);(A/name).write_bytes(raw)
print(r.stdout.strip())
