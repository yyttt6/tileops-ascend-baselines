"""Select only original-API CNN cases with completed v3 raw audit."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent; D=P/'vllm-fused5120-server1'
rs=json.loads((D/'RESULTS.json').read_text()); q=json.loads((P/'QUEUE.json').read_text()); out=[]
for r in rs:
 if r['variant']!='native' or r['status']!='formal_raw_verified': continue
 f=D/'remote/runs'/str(r['index'])/'native'/r['attempt']/'result.json'
 assert hashlib.sha256(f.read_bytes()).hexdigest()==r['result_sha256']
 source=json.loads((f.parent/'SOURCE.json').read_text())
 assert source['/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/vllm/basic.py']=='b4f860c870eec0c3991a1f5f0482fd531f9cb91871844c1e8c890d09ebb5aaf2'
 params=json.loads((f.parents[2]/'prepare.json').read_text())['params'];shape=params['output'][0]
 assert shape[-1]==5120
 row=next(x for x in q if (x['library'],x['op'],x['case'])==('vllm-ascend',r['op'],r['case']))
 ratio=r['us']/float(row['tileops_us'])
 reason='Retain original singleton addition plus original RMSNorm API, including input-dtype-rounded residual second output. Direct natural width5120 selects unsupported block2560; no kernel/compiler/block edits or relaxed precision. Current original full API15 raw windows audited, no optimization benefit claimed.'
 rec=dict(key=['vllm-ascend',r['op'],r['case']],index=r['index'],selected_variant='native',selected_us=r['us'],ratio=ratio,result_sha256=r['result_sha256'],decision='selected_under3' if ratio<3 else 'retain_original_api_over3',reason=reason)
 out.append(rec); row.update(review_status=rec['decision'],selected_us=r['us'],selected_ratio=ratio,decision_reason=reason,selected_evidence='VLLM_FUSED5120_SELECTION.json')
(P/'VLLM_FUSED5120_SELECTION.json').write_text(json.dumps(out,indent=2)+'\n')
(P/'QUEUE.json').write_text(json.dumps(q,indent=2)+'\n')
print([(r['key'],round(r['ratio'],3)) for r in out])
