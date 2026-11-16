from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;D=P/'sgl-layernorm-api-server1';T=P/'delivery/sgl_layernorm_api';T.mkdir(exist_ok=True);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();summary=json.loads((P/'SGL_LAYERNORM_API_DUAL_METRIC.partial.json').read_text());tasks=json.loads((D/'TASKS.json').read_text());routes=[]
for pair in summary['pairs']:
 for label in ['original','candidate']:
  r=pair[label];f=D/'remote/runs'/str(r['index'])/r['variant']/r['attempt']/'result.json';assert sha(f)==r['result_sha256'];v=json.loads(f.read_text());assert sha(f.parent/'SOURCE.json')==v['source_sha256'];src=json.loads((f.parent/'SOURCE.json').read_text())
  for file in [D/'native_adapters_v2.py',D/'original_layernorm.py',*(D/'source-review').glob('*.py'),*(D/'sgl-source').rglob('*.py')]:
   rel=file.relative_to(D);assert src['/opt/baseline-over3-20261110/'+D.name+'/'+str(rel)]==sha(file);dest=T/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
 task=tasks[pair['index']];routes.append({'key':[task['library'],task['op'],task['case']],'fixed':pair['candidate_device_us']<pair['original_device_us'],'evidence':pair,'historical_ratio_update':False})
(T/'EVIDENCE.json').write_text(json.dumps(summary,indent=2));(T/'ROUTES.json').write_text(json.dumps(routes,indent=2));print('staged',len(routes))
