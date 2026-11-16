"""Reuse prior complete audit only with immutable certificates and rehashed inputs."""
from pathlib import Path
import json,hashlib,shutil,datetime
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';CAT=D/'FLA_TAP_AUDIT_REUSE_CATALOG.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def load():
 return json.loads(CAT.read_text()) if CAT.exists() else {'certificates':{},'anchors':{}}
def promote(out,proof):
 c=load();new=[r for r in out['rows'] if str(r['ordinal']) not in c['certificates']]
 if not new:return
 digest=sha(proof);folder=A/'FLA_TAP_IMMUTABLE_AUDITS';folder.mkdir(exist_ok=True);anchor=folder/(digest+'.json')
 if not anchor.exists():shutil.copyfile(proof,anchor)
 assert sha(anchor)==digest;c['anchors'][digest]=str(anchor.relative_to(D))
 for r in new:
  cp=D/r['checkpoint'];assert sha(cp)==r['checkpoint_sha256'];e=json.loads(cp.read_text());assert e['guard']['performance_valid'] and e['guard']['phase']=='finished' and e['guard']['workload_exit_code']==0
  assert r['full_checker_15raw_guard_valid'] and r['original_Triton_source_recomputed_key_actual_loaded_ELF_launches_all15_verified']
  cap=D/'remote/runs'/e['tag']/str(r['ordinal'])/'JIT_CAPTURE';c['certificates'][str(r['ordinal'])]={'row':r,'anchor_sha256':digest,'retention_sha256':sha(cap/'RETENTION.json'),'source_sha256':e['batch']['source_sha256'],'archive_sha256':e['archive_sha256']}
 c['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=CAT.with_suffix('.tmp');tmp.write_text(json.dumps(c,indent=2)+chr(10));tmp.replace(CAT)
def verify(cp,e,a,cert,parser,root):
 assert sha(cp)==cert['row']['checkpoint_sha256'],'Cached checkpoint SHA changed'
 assert a['guard_valid'] and a['checkpoint_sha256']==cert['row']['checkpoint_sha256']
 assert e['archive_sha256']==cert['archive_sha256'] and e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0 and e['guard']['phase']=='finished'
 r=next(x for x in e['batch']['results'] if x['ordinal']==cert['row']['ordinal']);assert r['status']=='raw_checked_pending_guard' and any(x['ordinal']==r['ordinal'] and x['status']=='formal_raw_and_full_output_verified' for x in a['rows'])
 base=D/'remote/runs'/e['tag'];cap=base/str(r['ordinal'])/'JIT_CAPTURE';assert sha(base/'SOURCE.json')==cert['source_sha256']==e['batch']['source_sha256'];assert sha(cap/'RETENTION.json')==cert['retention_sha256'];ret=json.loads((cap/'RETENTION.json').read_text());assert ret['actual_source_cache_key_loaded_binary_launches_verified']
 for name,v in ret['files'].items():assert sha(cap/name)==v['sha256']
 original=root/'ascend_baselines/flash-linear-attention-npu/fla/ops/triton/triton_core/causal_conv1d.py'
 for k,rec in ret['records'].items():assert k==rec['hash']==rec['recomputed_hash'] and sha(cap/rec['source_copy'])==rec['source_sha256']==sha(root/'ascend_baselines/flash-linear-attention-npu/fla/ops/triton/triton_core'/Path(rec['source_file']).name)
 if r['key'][1] in ('SSDStatePassingFwdOp','SSDChunkCouplingFwdOp','SSDChunkCumsumFwdOp'):
  acap=cap/'AOT';assert sha(acap/'RETENTION.json')==ret['aot_retention_sha256'];ar=json.loads((acap/'RETENTION.json').read_text());assert ar['original_AoT_source_build_package_identity_verified']
  for name,v in ar['files'].items():assert sha(acap/name)==v['sha256']
  cpp=root/'ascend_baselines/flash-linear-attention-npu/fla/ops/ascendc'/('kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp' if r['key'][1]=='SSDChunkCumsumFwdOp' else 'gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/chunk_fwd_h.cpp')
  for rec in ar['records']:assert sha(cpp)==rec['original_CPP_sha256']
 t=r['timing'];assert t['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(t['retained_profiles'])==len(t['latency_samples_ms'])==15
 for prof in t['retained_profiles']:
  raw=D/'remote'/Path(prof['profile_path']).relative_to('/opt/baseline-over3-20261110/ordered100-20261112');assert parser.trace_digest(raw)==prof['profile_trace_sha256']
 return cert['row'],{'ordinal':r['ordinal'],'checkpoint_sha256':cert['row']['checkpoint_sha256'],'guard_valid':True,'dual_archive_sha256':e['archive_sha256'],'retention_sha256':cert['retention_sha256'],'reused_complete_all15_source_key_ELF_launch_profile_audit':True,'immutable_full_audit_anchor_sha256':cert['anchor_sha256'],'all15_trace_SHA_rechecked':True}
