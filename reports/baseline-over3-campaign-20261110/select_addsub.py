import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent;D=P/'sgl-addsub-server1';q=json.loads((P/'QUEUE.json').read_text());rs=json.loads((D/'RESULTS.json').read_text());selected=[]
for r in rs:
 if r['variant']!='sgl_fixed' or r['status']!='formal_raw_verified':continue
 controls=[x for x in rs if x['index']==r['index'] and x['variant']=='sgl_original' and x['status']=='formal_raw_verified'];assert len(controls)==1
 c=controls[0];proofs=[]
 for item in [c,r]:
  file=D/'remote/runs'/str(item['index'])/item['variant']/item['attempt']/'result.json';assert hashlib.sha256(file.read_bytes()).hexdigest()==item['result_sha256']
  proofs.append(json.loads((file.parent/'SOURCE.json').read_text()))
 assert proofs[0]==proofs[1]
 row=next(x for x in q if (x['library'],x['op'],x['case'])==('sgl-kernel-npu',r['op'],r['case']))
 chosen=r if r['us']<c['us'] else c;ratio=chosen['us']/float(row['tileops_us'])
 rec={'key':['sgl-kernel-npu',r['op'],r['case']],'index':r['index'],'control_us':c['us'],'candidate_us':r['us'],'selected_us':chosen['us'],'selected_variant':chosen['variant'],'selected_attempt':chosen['attempt'],'ratio':ratio,'old_ratio':float(row['ratio']),'result_sha256':chosen['result_sha256'],'windows':30,'decision':'selected_under3' if ratio<3 else 'selected_over3_pending_library_review','reason':'Skip pad only when tail=0. Original complete mul_add API and kernel unchanged; runtime broadcast materialization remains required by the flat equal-offset input reads.'}
 selected.append(rec);row.update(review_status=rec['decision'],selected_us=chosen['us'],selected_ratio=ratio,decision_reason=rec['reason'],selected_evidence='SGL_ADDSUB_SELECTION.partial.json')
(P/'SGL_ADDSUB_SELECTION.partial.json').write_text(json.dumps(selected,indent=2));(P/'QUEUE.json').write_text(json.dumps(q,indent=2));print([(x['key'][1:],round(x['ratio'],3)) for x in selected])
