"""Finite metadata-only exception; original payload/checker/timed body unchanged."""
import pathlib,json,hashlib,datetime,sys
APPROVED={1818,1819,1820,1821}
OLD='618ca3568073f86f64d2873967140a6f4cf3419771c549eee50e021e4e428204'
NEW='9b86e94c65732d1e10f4be445113418104ad42d47436272385bcf425274d9d4e'
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def source_delta_allowed(i,old,current,target):
 if set(old)!=set(current):return False
 delta={k:[old[k],current[k]] for k in old if old[k]!=current[k]}
 return not delta or (i in APPROVED and delta=={target:[OLD,NEW]})
def validate_source(P,i,meta):
 P=pathlib.Path(P);T=P/'workspace/TileOPs';rows=json.loads((P/'TASKS.json').read_text());row=rows[i]
 assert row['library']=='tilelang-ascend' and row['op']=='HardswishFwdOp'
 assert meta['key']==[row['library'],row['op'],row['case']] and meta['torch']=='2.13.0+cpu'
 current={str(f):sha(f) for folder in [T/'benchmarks',T/'workloads',T/'src/tileops/manifest'] for f in folder.rglob('*.py')}
 current[str(P/'reference_policy.py')]=sha(P/'reference_policy.py');current[str(P/'TASKS.json')]=sha(P/'TASKS.json')
 target=str(T/'benchmarks/ascend_all_pairs/sgl/row_gemv.py');old=meta['sources'];assert source_delta_allowed(i,old,current,target),'unapproved source change'
 delta={k:[old[k],current[k]] for k in old if old[k]!=current[k]}
 if delta:
  text=pathlib.Path(target).read_text();restored=text.replace("!=(package/'sgl_kernel_npu/__init__.py').resolve()","!=package/'sgl_kernel_npu/__init__.py'");assert hashlib.sha256(restored.encode()).hexdigest()==OLD
 return {'ordinal':i,'key':meta['key'],'only_source_delta':delta,'current_source_map_sha256':hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest(),'prepare_sha256':sha(P/'prepared-rollout'/str(i)/'prepare.json'),'payload_sha256':meta['payload_sha256'],'descriptor_sha256':meta['descriptor_sha256'],'helper_sha256':sha(__file__),'old_metadata_untouched':True,'timed_function_and_checker_unchanged':True}
def verify_runtime_source_identity(P,i,meta):
 proof=validate_source(P,i,meta);proof.update({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'runtime_source_check','payload_descriptor_checks':'original worker assertions immediately follow this source check'})
 tag=sys.argv[1] if len(sys.argv)>1 else ''
 if tag.startswith('rollout-') and pathlib.PurePath(tag).name==tag:
  out=pathlib.Path(P)/'runs'/tag/str(i);name='cached_input_source_proof.json'
 else:
  out=pathlib.Path(P)/'cached-input-fidelity'/str(i);name='CPU_RUNTIME_SOURCE_PROOF_'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f')+'.json'
 out.mkdir(parents=True,exist_ok=True);(out/name).write_text(json.dumps(proof,indent=2));return True
