from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;D=P/'vllm-rms-api-final-server1';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
rows=json.loads((D/'RESULTS.json').read_text());assert len(rows)==2 and all(r['status']=='formal_raw_verified' and r['windows']==15 for r in rows)
a,b=[next(r for r in rows if r['variant']==v) for v in ['native','fixed']]
paths=[D/'remote/runs/0'/v/'final1' for v in ['native','fixed']];sources=[json.loads((p/'SOURCE.json').read_text()) for p in paths]
assert all(sources[0][k]==sources[1][k] for k in sources[0].keys()&sources[1].keys())
for k in sources[0].keys()-sources[1].keys():assert k.endswith('/rmsnorm.py')
for k in sources[1].keys()-sources[0].keys():assert k.endswith('/layernorm_guard.py')
result={'scope':['vllm-ascend','RMSNormFwdOp',a['case']],'status':'current_environment_engineering_selection_verified_published_ratio_pending_environment_alignment','selected':'fixed','native_us':a['us'],'fixed_us':b['us'],'reduction_pct':100*(1-b['us']/a['us']),'protocol':'original isolated15 both sides','environment':json.loads((paths[0]/'result.json').read_text())['hardware'],'source_common_equal':True,'source_difference':'two intended complete vendor API source paths and their generated import bindings','reason':'Two original complete APIs measured; both one task per call, zero AICPU. No wrapper loop removed. Alternate API modestly faster, selected without kernel changes; no claim all32 cases improve.','historical_ratio':44.258375666788034,'published_ratio_updated':False,'ratio_caveat':'historical Torch2.10 vs measured2.13; current denominator not remeasured. Do not present hybrid ratio as current accepted speedup.','evidence':rows}
(P/'VLLM_RMS_API_SELECTION.partial.json').write_text(json.dumps(result,indent=2))
T=P/'delivery/vllm_rms_api';T.mkdir(exist_ok=True)
for n in ['native_adapters_v2.py','normalization_layout_candidate.py']:
 f=D/n
 for src in sources:assert src['/opt/baseline-over3-20261110/'+D.name+'/'+n]==sha(f)
 (T/n).write_bytes(f.read_bytes())
(T/'SELECTION.json').write_text(json.dumps(result,indent=2));(T/'DELIVERY.txt').write_text('Staged exact representative only; use factory(...,fixed=True,case=case). Original runtime source SHA assertion preserved. No shared registry/web installation. Current environment isolated15 each verified; historical denominator/environment alignment remains pending.\n')
print(json.dumps({k:result[k] for k in ['status','native_us','fixed_us','reduction_pct']}))
