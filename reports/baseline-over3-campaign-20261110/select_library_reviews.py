from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;q=json.loads((P/'QUEUE.json').read_text())
for sub,lib,report in [('sgl-mean-api-server1','sgl-kernel-npu','SGL_MEAN_SELECTION.json'),('deepselect-reductions-server2','DeepSelect','DEEPSELECT_REDUCTION_SELECTION.partial.json')]:
 d=P/sub;out=[]
 for r in json.loads((d/'RESULTS.json').read_text()):
  if r['status']!='formal_raw_verified' or r['variant'] not in ['native','sgl_original']:continue
  f=d/'remote/runs'/str(r['index'])/r['variant']/r['attempt']/'result.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==r['result_sha256']
  key=[lib,r['op'],r['case']];row=next(x for x in q if [x['library'],x['op'],x['case']]==key);ratio=r['us']/float(row['tileops_us'])
  pending=sub.startswith('deepselect') and r['index'] in [1,2]
  if sub.startswith('sgl'):
   reason='Existing complete SIMD mean API freshly measured with original checker. Alternate complete API failed compilation for representative BF16 width4096 with UB overflow; other2 alternate cases not separately attempted or claimed failed. Original kernel math/schedule unchanged. No obvious redundant per-call allocation in original provider; shape-only ones already allocated at factory. Retain existing implementation within no-kernel-tuning scope, not a proof of optimal performance.'
  else:
   reason='Original complete TopK adapter freshly measured and raw-audited. Preserve full reduction/window packing, private aligned input storage and original32768 row limit. Public API offers TopK, not dedicated max pooling/reduction. Global adaptive cases have one window. Prior representative copy-only test negligible and largebatch failed; no new kernel or relaxed contract.'
   if pending:reason='Current complete API verified; equal-width adaptive-window grouping candidate prepared, not measured. Retain pending review status.'
  status='current_api_verified_pending_candidate' if pending else ('selected_under3' if ratio<3 else 'retain_original_over3_with_reason')
  rec=dict(key=key,index=r['index'],selected_variant=r['variant'],selected_us=r['us'],ratio=ratio,result_sha256=r['result_sha256'],windows=15,decision=status,reason=reason);out.append(rec)
  if pending:
   if row.get('selected_evidence')!='DEEPSELECT_ADAPTIVE_SELECTION.json':row.update(review_status=status,current_verified_us=r['us'],current_verified_ratio=ratio,current_evidence=report)
  else:row.update(review_status=status,selected_us=r['us'],selected_ratio=ratio,decision_reason=reason,selected_evidence=report)
 (P/report).write_text(json.dumps(out,indent=2)+'\n');print(report,len(out))
(P/'QUEUE.json').write_text(json.dumps(q,indent=2)+'\n')
