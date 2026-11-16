import json,pathlib,hashlib
P=pathlib.Path(__file__).resolve().parent;D=P/'sgl-pool-api-server1';q=json.loads((P/'QUEUE.json').read_text());rs=json.loads((D/'RESULTS.json').read_text());first=json.loads((P/'sgl-pool-server1/RESULTS.json').read_text());out=[]
for r in rs:
 if r['variant']!='sgl_fixed' or r['status']!='formal_raw_verified':continue
 c=next(x for x in rs if x['index']==r['index'] and x['variant']=='sgl_original' and x['status']=='formal_raw_verified');proofs=[]
 for x in [c,r]:
  f=D/'remote/runs'/str(x['index'])/x['variant']/x['attempt']/'result.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==x['result_sha256'];proofs.append(json.loads((f.parent/'SOURCE.json').read_text()))
 assert proofs[0]==proofs[1]
 old=next(x for x in first if x['index']==r['index'] and x['variant']=='sgl_original' and x['status']=='formal_raw_verified')
 row=next(x for x in q if [x['library'],x['op'],x['case']]==['sgl-kernel-npu',r['op'],r['case']]);selected=min([c,r],key=lambda x:x['us']);ratio=selected['us']/float(row['tileops_us'])
 rec=dict(key=['sgl-kernel-npu',r['op'],r['case']],index=r['index'],selected_variant=selected['variant'],selected_us=selected['us'],ratio=ratio,original_per_chunk_us=old['us'],batched_simd_us=c['us'],batched_alternate_api_us=r['us'],result_sha256=selected['result_sha256'],decision='selected_under3' if ratio<3 else 'selected_over3_pending_boundary_review',reason='Batch independent complete rows and select existing full layer_norm_fwd_npu API mean output. Vendor kernel/config/math unchanged; full transpose, weight allocation, unused normalization work and output cast remain timed. Original checker and precision unchanged.')
 out.append(rec);row.update(review_status=rec['decision'],selected_us=selected['us'],selected_ratio=ratio,decision_reason=rec['reason'],selected_evidence='SGL_POOL_SELECTION.partial.json')
(P/'SGL_POOL_SELECTION.partial.json').write_text(json.dumps(out,indent=2)+'\n');(P/'QUEUE.json').write_text(json.dumps(q,indent=2)+'\n');print([(x['key'][2],round(x['ratio'],3)) for x in out])
