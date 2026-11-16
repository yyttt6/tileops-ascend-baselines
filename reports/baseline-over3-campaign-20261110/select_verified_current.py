from pathlib import Path
import json,hashlib
p=Path('reports/baseline-over3-campaign-20261110');q=json.loads((p/'QUEUE.json').read_text());out=[]
for x in json.loads((p/'SGL_ELEMENTWISE_CURRENT.json').read_text()):
 x=dict(x);x['decision']='retain_original_over3_with_reason';x['reason']='Measured complete fused_scale_shift API has independent scale values, requiring hidden_size=numel and num_tokens=1; its native tiling processes blocks of64/80/96 rows although only one is valid. Shared-scale reshape is invalid for these inputs. Reviewed mul_add uses scalar constexpr factor, sigmoid_mul changes semantics, rsqrt_mul introduces normalization/shared weights; none supplies a direct arbitrary tensor product replacement. Scalar-shift scale_shift branch hardcodes1+scale and cannot substitute fullzero shift without changing arithmetic. Retain current complete API under no-kernel-optimization scope; not a universal lower bound. All15 windows have zero AICPU.'
 x['selected_us']=x['us'];x['selected_variant']='sgl_original';out.append(x)
 row=next(r for r in q if [r['library'],r['op'],r['case']]==x['key']);row.update(review_status=x['decision'],selected_us=x['us'],selected_ratio=x['ratio'],decision_reason=x['reason'],selected_evidence='SGL_ELEMENTWISE_SELECTION.json')
(p/'SGL_ELEMENTWISE_SELECTION.json').write_text(json.dumps(out,indent=2)+'\n')
d=p/'deepselect-server2';rs=json.loads((d/'RESULTS.json').read_text());out=json.loads((p/'DEEPSELECT_SELECTION.partial.json').read_text());keys={tuple(x['key']) for x in out}
for x in rs:
 if x['status']!='formal_raw_verified' or x['variant']!='native':continue
 key=['DeepSelect',x['op'],x['case']]
 if tuple(key) in keys:continue
 f=d/'remote/runs'/str(x['index'])/'native'/x['attempt']/'result.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==x['result_sha256']
 row=next(r for r in q if [r['library'],r['op'],r['case']]==key);ratio=x['us']/float(row['tileops_us'])
 rec=dict(key=key,selected_variant='native',selected_us=x['us'],ratio=ratio,result_sha256=x['result_sha256'],windows=15,decision='retain_original_over3_with_reason',reason='Current original complete TopK API measured and raw-audited for this exact case. Library public interface supplies TopK, not pairwise maximum. Representative direct packing removed one copy with only0.08% difference; larger batch failed and is not adopted. No per-case candidate claim; retain original library adapter without writing a new kernel.')
 out.append(rec);row.update(review_status=rec['decision'],selected_us=x['us'],selected_ratio=ratio,decision_reason=rec['reason'],selected_evidence='DEEPSELECT_SELECTION.partial.json')
(p/'DEEPSELECT_SELECTION.partial.json').write_text(json.dumps(out,indent=2)+'\n');(p/'QUEUE.json').write_text(json.dumps(q,indent=2)+'\n');print('DeepSelect selected',len(out))
