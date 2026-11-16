"""Close historical build/adapter fidelity without repeating 41 valid measurements."""
import pathlib,json,hashlib,importlib.util,statistics,ast,time
D=pathlib.Path(__file__).resolve().parent;P=D.parent;A=P/'atvc-server1';REC=P/'vendor-runtime-repair/atvc-recovered'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
build=json.loads((REC/'addcmul-rows8-build.json').read_text());recovery=json.loads((REC/'RECOVERY.json').read_text());effective={};diff=[]
for key,digest in build['source_sha256'].items():
 recovered=REC/'source'/digest/key;assert recovered.is_file() and sha(recovered)==digest,(key,digest)
 effective[key]={'sha256':digest,'recovered':str(recovered)}
 if key.startswith('ascend_baselines/atvc/'):
  current=pathlib.Path(key)
  if sha(current)!=digest:diff.append((key,recovered,current))
assert len(diff)==1 and diff[0][0].endswith('include/common/compile_info.h'),diff
key,recovered,current=diff[0]
branch='    if (soc == platform_ascendc::SocVersion::ASCEND950) {\n        uint64_t ubSize = 0;\n        platformInfoMgr->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);\n        return {platformInfoMgr->GetCoreNumAiv(), ubSize, 256, 32};\n    }\n'
assert recovered.read_text().count(branch)==1 and recovered.read_text().replace(branch,'')==current.read_text()
bridge=effective['TileOPs/benchmarks/ascend_all_pairs/atvc/addcmul_bridge.cpp'];bridgefile=pathlib.Path(bridge['recovered'])
assert sha(A/'source-review/addcmul_bridge.cpp')==bridge['sha256']
s=bridgefile.read_text();assert 'param.nBufferNum = 1' in s and '{1, count}, {8, count}' in s and 'CalcBroadcastTiling<Traits>' in s
assert 'param.nBufferNum = 1' in pathlib.Path('ascend_baselines/atvc/examples/addcmul/addcmul.cpp').read_text()
# No new mathematical/tiling kernel change: both original BroadcastPolicy paths,
# full vendor AddcmulCustom and original Traits remain in the captured C ABI.
assert s.count('AddcmulCustom<Traits, ATVC::BROADCAST_POLICY')==2
adapter=A/'source-review/addcmul.py';tree=ast.parse(adapter.read_text());run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run');timed=ast.unparse(run)
assert 'base.copy_' in timed and 'pack(value)' in timed and 'lib.atvc_addcmul_run' in timed and 'output[0].reshape(shape).to(output_dtype)' in timed
parserfile=D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py';sp=importlib.util.spec_from_file_location('parser',parserfile);parser=importlib.util.module_from_spec(sp);sp.loader.exec_module(parser)
known=json.loads((P/'ATVC_CURRENT.partial.json').read_text());tasks=json.loads((D/'TASKS.json').read_text());bykey={tuple(x['key']):x for x in known};rows=[];t=time.perf_counter()
for r in tasks:
 k=(r['library'],r['op'],r['case'])
 if k not in bykey:continue
 old=bykey[k];folder=A/'remote/runs'/str(old['index'])/'native/f1';resultfile=folder/'result.json';m=json.loads(resultfile.read_text());assert sha(resultfile)==old['result_sha256']
 source=json.loads((folder/'SOURCE.json').read_text());assert sha(folder/'SOURCE.json')==m['source_sha256']
 binarykey='/opt/tileops-all-pairs/workspace/.cache/all-pairs/atvc/libatvc_addcmul_rows8.so';assert source[binarykey]==build['binary_sha256']
 assert source['/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/atvc/addcmul_bridge.cpp']==bridge['sha256']
 assert source['/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/atvc/addcmul.py']==sha(adapter)
 guardpath=A/'remote'/('formal-'+str(old['index'])+'-native-f1-guard.json');g=json.loads(guardpath.read_text());assert g['performance_valid'] and g['phase']=='finished' and g['workload_exit_code']==0
 check=json.loads((folder/'correctness.json').read_text());assert sha(folder/'correctness.json')==m['correctness_sha256'];assert check['inputs_preserved'] and check['check']['checked_outputs']==check['check']['total_outputs']==1 and not check['check']['unchecked_reason']
 times=[]
 for prof in m['timing']['retained_profiles']:
  raw=A/'remote'/pathlib.Path(prof['profile_path']).relative_to('/opt/baseline-over3-20261110/atvc-server1');assert parser.trace_digest(raw)==prof['profile_trace_sha256'];piece=parser.parse_profile(raw,1);assert piece['profile_named_task_coverage_complete'];times.append(piece['latency_ms'])
 assert len(times)==15 and all(abs(a-b)<1e-9 for a,b in zip(times,m['timing']['latency_samples_ms'])) and abs(statistics.mean(times)-m['timing']['latency_ms'])<1e-9
 rows.append({'ordinal':r['ordinal'],'key':k,'status':'historical_valid_measurement_reused_after_build_and_adapter_review','device_us':m['timing']['latency_ms']*1000,'old_result_sha256':sha(resultfile),'guard_sha256':sha(guardpath),'fresh_device_windows':0,'replayed_historical_windows':15,'kernel_and_both_broadcast_policies_unchanged':True,'selection':'retain_original_complete_api_over3','reason':'original Addcmul example full computation and eight-row broadcast/packs remain; source/header gap now resolved, no native API-only improvement validated; unsupported natural-layout candidate is not adopted'})
assert len(rows)==41
report={'status':'41_historical_build_and_adapter_fidelity_reviews_complete','rows':rows,'effective_build_sources':effective,'captured_live_header_differed_from_actually_built_header':True,'supplementary_historical_platform_patch':{'file':key,'change':'add ASCEND950 recognition and query actual AIV count/UB bytes; rest byte-identical to locked header','not_a_kernel_or_tuning_change':True,'built_sha256':sha(recovered),'live_locked_sha256':sha(current)},'binary_sha256':build['binary_sha256'],'original_measurements_unchanged':True,'fresh_measurement_count':0,'seconds':time.perf_counter()-t}
(D/'ATVC_REUSE_REVIEW.json').write_text(json.dumps(report,indent=2));print('REUSED',len(rows),'old windows',len(rows)*15,'CPU seconds',report['seconds'])
