from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).parent;D=P/'vllm-softmax-matched-server1';T=P/'delivery/vllm_softmax_matched';T.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=json.loads((D/'RESULTS.json').read_text()) if (D/'RESULTS.json').exists() else [];tasks=json.loads((D/'TASKS.json').read_text());good=[]
for row in rows:
 if row['status']!='formal_raw_verified':continue
 f=D/'remote/runs'/str(row['index'])/row['variant']/row['attempt']/'result.json';m=json.loads(f.read_text());assert sha(f)==row['result_sha256'];assert sha(f.parent/'SOURCE.json')==m['source_sha256'];source=json.loads((f.parent/'SOURCE.json').read_text());assert m['hardware']['torch']=='2.10.0+cpu' and m['hardware']['torch_npu']=='2.10.0.post4'
 adapter=D/'native_adapters_v2.py';assert source['/opt/baseline-over3-20261110/'+D.name+'/native_adapters_v2.py']==sha(adapter)
 good.append({'key':[tasks[row['index']]['library'],row['op'],row['case']],'selected':'original','evidence':row,'runtime':m['hardware'],'historical_ratio_update':False,'reason':'Complete original moe_gating_top_k API exposes full Softmax scores; extra TopK plus all casts/packing retained; no candidate benefit claimed'})
shutil.copyfile(D/'native_adapters_v2.py',T/'native_adapters_v2.py');summary={'status':'incremental original API isolated15 raw/checker/guard/source verified; historical denominator provenance separate','valid':len(good),'total':12,'formal_windows':15*len(good),'routes':good};(P/'VLLM_SOFTMAX_MATCHED_CURRENT.json').write_text(json.dumps(summary,indent=2));(T/'EVIDENCE.json').write_text(json.dumps(summary,indent=2));(T/'ROUTES.json').write_text(json.dumps(good,indent=2));print('Softmax staged',len(good),'of12')
