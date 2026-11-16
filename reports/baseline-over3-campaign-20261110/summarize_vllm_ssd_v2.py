from pathlib import Path
import json,hashlib,ast,shutil
P=Path(__file__).resolve().parent;O=P/'vllm-ssd-coupling-server1';D=P/'vllm-ssd-coupling-v2-server1';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=next(x for x in json.loads((O/'RESULTS.json').read_text()) if x['variant']=='native' and x['index']==0 and x['status']=='formal_raw_verified');b=next(x for x in json.loads((D/'RESULTS.json').read_text()) if x['variant']=='fixed' and x['index']==0 and x['status']=='formal_raw_verified');paths=[O/'remote/runs/0/native/final1',D/'remote/runs/0/fixed/final1'];ms=[];sources=[]
for path,row in zip(paths,[a,b]):
 assert sha(path/'result.json')==row['result_sha256'];m=json.loads((path/'result.json').read_text());assert sha(path/'SOURCE.json')==m['source_sha256'];ms.append(m);sources.append(json.loads((path/'SOURCE.json').read_text()))
p0=json.loads((O/'remote/runs/0/prepare.json').read_text());p1=json.loads((D/'remote/runs/0/prepare.json').read_text());assert p0==p1,'input/checker preparation changed'
h0=ms[0]['hardware'].copy();h1=ms[1]['hardware'].copy();h0.pop('backend');h1.pop('backend');assert h0==h1
external=[{k:v for k,v in s.items() if '/opt/baseline-over3-20261110/' not in k} for s in sources];assert external[0]==external[1]
internal=[];resolved=[]
for base,src in zip([O,D],sources):
 prefix='/opt/baseline-over3-20261110/'+base.name+'/'
 sub={k[len(prefix):]:v for k,v in src.items() if k.startswith(prefix)}
 files={}
 for fn,digest in sub.items():
  options=[base/fn,*base.glob(fn+".before-expand-*")]
  files[fn]=next(p for p in options if p.exists() and sha(p)==digest)
 resolved.append(files)
 internal.append(sub)
assert internal[0].keys()==internal[1].keys()
changed=[]
for fn in internal[0]:
 old=resolved[0][fn].read_text();new=resolved[1][fn].read_text().replace(D.name,O.name)
 if fn=='ssd_candidate.py':
  new=new.replace("torch.tril_indices(q, q, device='cpu').to(c_mat.device)",'torch.tril_indices(q, q, device=c_mat.device)')
  changed.append(fn+': percall CPU integer indices and transfer, all timed')
 if fn=='run_expansion.py':
  old=old.replace("  run(i,'native','final1','formal')\n",'')
  changed.append(fn+': skip already valid original control, only new candidate dispatched')
 assert new==old,fn
for fn in ['coupling_original','ssd_original']:
 def node(base):return ast.dump(next(x for x in ast.parse((base/'ssd_candidate.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name==fn))
 assert node(O)==node(D)
proof={'original_control_reused':True,'not_full_source_identical':True,'prepare_full_equal':True,'payload_sha256':p0['payload_sha256'],'hardware_equal_except_expected_backend_label':h0,'external_source_files_equal':len(external[0]),'internal_source_normalized_equal_except_candidate':changed,'original_functions_ast_equal':True,'original_result_sha256':a['result_sha256'],'v2_result_sha256':b['result_sha256']}
pair={'key':['vllm-ascend',a['op'],a['case']],'original':a,'candidate_v2':b,'host_reduction_pct':100*(1-b['host_us']/a['host_us']),'device_change_pct':100*(b['us']/a['us']-1),'proof':proof,'status':'dualmetrics audited; old original control reused after explicit identity proof, no historical ratio update'}
(P/'VLLM_SSD_V2_DUAL_METRIC.partial.json').write_text(json.dumps({'pairs':[pair]},indent=2)+'\n');(D/'CONTROL_REUSE_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
T=P/'delivery/vllm_ssd_coupling_dual_metric';T.mkdir(exist_ok=True)
for fn in ['native_adapters_v2.py','ssd_candidate.py']:shutil.copy2(D/fn,T/fn)
(T/'EVIDENCE.json').write_text(json.dumps(pair,indent=2)+'\n');(T/'DELIVERY.txt').write_text('Both wrappers staged, original control explicitly reused after full input/verification, external source and normalized internal source identity checks. Host/device must remain separate; no automatic install or historical ratio update. V1 regression retained separately.\n')
print(pair['host_reduction_pct'],pair['device_change_pct'],proof['external_source_files_equal'])
