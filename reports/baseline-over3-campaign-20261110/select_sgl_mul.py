import json,pathlib,hashlib
P=pathlib.Path(__file__).resolve().parent;q=json.loads((P/'QUEUE.json').read_text());old=P.parent/'high-speedup-baseline-fix-server2-20261110';selected=[]
audit=json.loads((old/'SOURCE_IDENTITY_AUDIT.json').read_text());assert next(a for a in audit if (a['index'],a['variant'],a['attempt'])==(1,'sgl_fixed','layout_f1'))['status']=='matched'
base=json.loads((old/'remote/runs/1/sgl_fixed/layout_f1/SOURCE.json').read_text())
for root,origin in [(old,'reused_formal'),(P/'sgl-server1','new_formal')]:
 rs=json.loads((root/'RESULTS.json').read_text())
 for r in rs:
  if r['variant']!='sgl_fixed' or r['status']!='formal_raw_verified' or r['op']!='MulFwdOp':continue
  file=root/'remote/runs'/str(r['index'])/r['variant']/r['attempt']/'result.json';assert hashlib.sha256(file.read_bytes()).hexdigest()==r['result_sha256']
  proof=json.loads((file.parent/'SOURCE.json').read_text())
  if origin=='reused_formal':assert all(proof.get(k)==v for k,v in base.items()),'previous source audit no longer reusable'
  row=next(x for x in q if (x['library'],x['op'],x['case'])==('sgl-kernel-npu',r['op'],r['case']))
  controls=[x for x in rs if x['index']==r['index'] and x['variant']=='sgl_original' and x['status']=='formal_raw_verified'];assert len(controls)==1
  cf=root/'remote/runs'/str(controls[0]['index'])/'sgl_original'/controls[0]['attempt']/'result.json';assert hashlib.sha256(cf.read_bytes()).hexdigest()==controls[0]['result_sha256']
  if origin=='reused_formal':
   cp=json.loads((cf.parent/'SOURCE.json').read_text());assert all(cp.get(k)==v for k,v in base.items())
  ratio=r['us']/float(row['tileops_us']);record={'key':['sgl-kernel-npu',r['op'],r['case']],'origin':origin,'selected_us':r['us'],'control_us':controls[0]['us'],'ratio':ratio,'old_ratio':float(row['ratio']),'windows':30,'source_identity':'all previously audited source entries equal, including original controls' if origin=='reused_formal' else 'immutable isolated source; worker runtime source proof unchanged','result_sha256':r['result_sha256'],'evidence':str(file),'decision':'selected_under3' if ratio<3 else 'selected_improved_over3_pending_boundary_review'}
  selected.append(record);row.update(review_status=record['decision'],selected_us=r['us'],selected_ratio=ratio,decision_reason='Original complete SGL fused_scale_shift API with broadcast-preserving layout; complete wrapper timed; no kernel changes.',selected_evidence=str(file))
(P/'SGL_MUL_SELECTION.json').write_text(json.dumps(selected,indent=2));(P/'QUEUE.json').write_text(json.dumps(q,indent=2))
print([(r['key'][2],round(r['ratio'],3)) for r in selected])
