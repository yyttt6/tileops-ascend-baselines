from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;D=P/'vllm-fused-server1';rs=json.loads((D/'RESULTS.json').read_text());q=json.loads((P/'QUEUE.json').read_text());out=[]
for r in rs:
 if r['variant']!='fixed' or r['status']!='formal_raw_verified':continue
 c=next(x for x in rs if x['index']==r['index'] and x['variant']=='native' and x['status']=='formal_raw_verified');proof=[]
 for x in [c,r]:
  f=D/'remote/runs'/str(x['index'])/x['variant']/x['attempt']/'result.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==x['result_sha256'];proof.append(json.loads((f.parent/'SOURCE.json').read_text()))
 assert proof[0]==proof[1]
 row=next(x for x in q if (x['library'],x['op'],x['case'])==('vllm-ascend',r['op'],r['case']));chosen=min([r,c],key=lambda x:x['us']);ratio=chosen['us']/float(row['tileops_us'])
 rec=dict(key=['vllm-ascend',r['op'],r['case']],index=r['index'],selected_us=chosen['us'],selected_variant=chosen['variant'],control_us=c['us'],candidate_us=r['us'],ratio=ratio,result_sha256=chosen['result_sha256'],decision='selected_under3' if ratio<3 else 'retain_best_engineering_layout_over3',reason='Keep natural hidden-feature layout for original addition API, retaining original RMSNorm and input-dtype-rounded sum as second output. Full packing/casts/allocations timed; no kernel/config/checker changes. Original/control each15 raw windows audited.');out.append(rec);row.update(review_status=rec['decision'],selected_us=chosen['us'],selected_ratio=ratio,decision_reason=rec['reason'],selected_evidence='VLLM_FUSED_SELECTION.partial.json')
(P/'VLLM_FUSED_SELECTION.partial.json').write_text(json.dumps(out,indent=2)+'\n');(P/'QUEUE.json').write_text(json.dumps(q,indent=2)+'\n');print([(x['key'][2],round(x['ratio'],3)) for x in out])
