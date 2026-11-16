from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;D=P/'sgl-hardsigmoid-current-server1';T=P/'delivery/sgl_hardsigmoid_current';T.mkdir(exist_ok=True);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();out=[];tasks=json.loads((D/'TASKS.json').read_text())
for r in json.loads((D/'RESULTS.json').read_text()):
 if r['variant']!='native' or r['status']!='formal_raw_verified':continue
 d=D/'remote/runs'/str(r['index'])/r['variant']/r['attempt'];m=json.loads((d/'result.json').read_text());assert sha(d/'result.json')==r['result_sha256'];assert sha(d/'SOURCE.json')==m['source_sha256'];src=json.loads((d/'SOURCE.json').read_text())
 for f in [D/'native_adapters_v2.py',D/'original_hardsigmoid.py',D/'original_sub_body.py',*(D/'sgl-source').rglob('*.py'),*(D/'source-review').glob('*.py')]:
  rel=f.relative_to(D);assert src['/opt/baseline-over3-20261110/'+D.name+'/'+str(rel)]==sha(f);dest=T/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
 t=tasks[r['index']];out.append({'key':[t['library'],t['op'],t['case']],'fixed':False,'evidence':r,'hardware':m['hardware'],'status':'originalAPIverified_noimprovementclaim','historical_ratio_update':False})
(P/'SGL_HARDSIGMOID_CURRENT.partial.json').write_text(json.dumps({'pairs':out},indent=2)+'\n');(T/'EVIDENCE.json').write_text(json.dumps(out,indent=2)+'\n');print('verified',len(out))
