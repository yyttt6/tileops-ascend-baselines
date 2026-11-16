from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;D=P/'sgl-rms-current-server1';T=P/'delivery/sgl_rms_current';T.mkdir(exist_ok=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
rows=json.loads((D/'RESULTS.json').read_text()) if (D/'RESULTS.json').exists() else [];tasks=json.loads((D/'TASKS.json').read_text());out=[]
for r in rows:
 if r['status']!='formal_raw_verified':continue
 d=D/'remote/runs'/str(r['index'])/r['variant']/r['attempt'];m=json.loads((d/'result.json').read_text());assert sha(d/'result.json')==r['result_sha256'];assert sha(d/'SOURCE.json')==m['source_sha256'];source=json.loads((d/'SOURCE.json').read_text())
 for f in [D/'native_adapters_v2.py',D/'provider.py',*(D/'sgl-source').rglob('*.py')]:
  rel=f.relative_to(D);assert source['/opt/baseline-over3-20261110/'+D.name+'/'+str(rel)]==sha(f);dst=T/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
 t=tasks[r['index']];out.append({'key':[t['library'],t['op'],t['case']],'evidence':r,'hardware':m['hardware'],'status':'original_complete_API_verified_current_environment_no_optimization_claim','historical_ratio_update':False})
result={'pairs':out,'original_isolated_windows':15*len(out),'historical_ratio_update':False};(P/'SGL_RMS_CURRENT.partial.json').write_text(json.dumps(result,indent=2)+'\n');(T/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n');(T/'DELIVERY.txt').write_text('Original complete SGL RMSNorm API, independent original Triton files only; no kernel edits, no optimization claim. Exact evidence keys only, not installed globally. Private package path must be rebound on integration.\n');print('verified',len(out))
