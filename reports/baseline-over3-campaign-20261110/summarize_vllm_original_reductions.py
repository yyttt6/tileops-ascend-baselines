from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pairs=[];pending=[]
for name,adapter in [('vllm-mean-current-server1','mean_api_copy.py'),('vllm-layernorm-current-server2','normalization_api_copy.py'),('vllm-groupnorm-current-server1','groupnorm_api_copy.py'),('vllm-groupnorm-current-server2','groupnorm_api_copy.py')]:
 D=P/name;tasks=json.loads((D/'TASKS.json').read_text());rows=json.loads((D/'RESULTS.json').read_text()) if (D/'RESULTS.json').exists() else []
 T=P/'delivery'/name;T.mkdir(exist_ok=True)
 for n in ['native_adapters_v2.py',adapter]:(T/n).write_bytes((D/n).read_bytes())
 own=[]
 for i,t in enumerate(tasks):
  r=next((r for r in rows if r['index']==i and r['variant']=='native' and r['attempt']=='final1' and r['status']=='formal_raw_verified'),None)
  if r is None:pending.append([name,i,t['case']]);continue
  d=D/'remote/runs'/str(i)/'native/final1';m=json.loads((d/'result.json').read_text());assert sha(d/'result.json')==r['result_sha256'];assert sha(d/'SOURCE.json')==m['source_sha256'];source=json.loads((d/'SOURCE.json').read_text())
  for n in ['native_adapters_v2.py',adapter]:assert source['/opt/baseline-over3-20261110/'+name+'/'+n]==sha(D/n)
  item={'key':[t['library'],t['op'],t['case']],'batch':name,'index':i,'isolated_us':r['us'],'evidence':r,'hardware':m['hardware'],'source_verified':True,'status':'current_environment_original_API_verified_no_improvement_claim','historical_ratio_update':False}
  pairs.append(item);own.append(item)
 (T/'SELECTION.partial.json').write_text(json.dumps(own,indent=2))
 (T/'ROUTES.partial.json').write_text(json.dumps([{'key':x['key'],'fixed':False} for x in own],indent=2))
out={'pairs':pairs,'pending':pending,'isolated_windows':15*len(pairs),'metric':'device; wall diagnostic separate','historical_ratio_update':False}
(P/'VLLM_ORIGINAL_REDUCTIONS_CURRENT.partial.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'complete':len(pairs),'pending':len(pending),'pairs':[(x['key'][1:],x['isolated_us']) for x in pairs]}))
