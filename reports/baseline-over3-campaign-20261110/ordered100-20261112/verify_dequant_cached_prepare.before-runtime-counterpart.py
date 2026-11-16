"""Exact-case cached input reuse; generic prepare gate unchanged, no Torch/NPU import."""
import pathlib,json,sys,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';i=int(sys.argv[1]);allow={986,987,988,989,979,980,981,982,983,984,985};assert i in allow,'explicit scope only'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as fd:
  for b in iter(lambda:fd.read(8388608),b''):h.update(b)
 return h.hexdigest()
record=P/'prepared-rollout'/str(i);meta=json.loads((record/'prepare.json').read_text());rows=json.loads((P/'TASKS.json').read_text());row=rows[i];assert meta['key']==[row['library'],row['op'],row['case']] and row['library']=='sgl-kernel-npu' and row['op'] in ['INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp'];assert meta['torch']=='2.10.0+cpu'
source={str(f):sha(f) for folder in [T/'benchmarks',T/'workloads',T/'src/tileops/manifest'] for f in folder.rglob('*.py')};source[str(P/'reference_policy.py')]=sha(P/'reference_policy.py');source[str(P/'TASKS.json')]=sha(P/'TASKS.json');old=meta['sources'];assert set(old)==set(source),'added/removed source requires new review'
changed={k:[old[k],source[k]] for k in old if old[k]!=source[k]};target=str(T/'benchmarks/ascend_all_pairs/sgl/row_gemv.py');expected={target:['618ca3568073f86f64d2873967140a6f4cf3419771c549eee50e021e4e428204','9b86e94c65732d1e10f4be445113418104ad42d47436272385bcf425274d9d4e']};assert changed==expected,'any other source change rejected'
text=pathlib.Path(target).read_text();restored=text.replace("!=(package/'sgl_kernel_npu/__init__.py').resolve()","!=package/'sgl_kernel_npu/__init__.py'");assert hashlib.sha256(restored.encode()).hexdigest()==expected[target][0],'path-only proof failed'
assert sha(record/'payload.pt')==meta['payload_sha256'] and sha(record/'case.pkl')==meta['descriptor_sha256'],'unchanged full input/descriptor identity required'
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':i,'key':meta['key'],'cached_prepare_sha256':sha(record/'prepare.json'),'payload_sha256':meta['payload_sha256'],'descriptor_sha256':meta['descriptor_sha256'],'only_source_delta':changed,'all_other_sources_identical':len(source)-1,'current_source_map_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),'reverse_path_change_restores_entire_file_SHA':True,'old_metadata_untouched':True,'no_Torch_or_NPU_imported':True}
out=P/'cached-input-fidelity'/str(i);out.mkdir(parents=True,exist_ok=True);(out/'PROOF.json').write_text(json.dumps(proof,indent=2));print('FROZEN_DEQUANT_CACHED_INPUT_VERIFIED',i,proof['cached_prepare_sha256'],flush=True)
