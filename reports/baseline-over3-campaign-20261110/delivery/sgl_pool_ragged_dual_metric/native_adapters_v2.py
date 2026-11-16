import pathlib,sys,hashlib
P=pathlib.Path(__file__).resolve().parent
from sgl_layout_v1 import ROOT
def factory(op,params,fixed,args,*,case):
 assert op=='MeanPoolingFwdOp' and case.id.startswith('ragged')
 assert hashlib.sha256((P/'additional.py').read_bytes()).hexdigest()=='aab097f709da3cb91bce37e73ef6a482ed78a0b891ba0c8ef645d71f337508d8'
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 sys.path.insert(0,str(ROOT))
 from sgl_namespace_v1 import initialize,bindings
 initialize()
 from sgl_kernel_npu.fla.layernorm_gated import layer_norm_fwd_npu_simd,layer_norm_fwd_npu
 from api_only_candidate import build
 fn=build(layer_norm_fwd_npu if fixed else layer_norm_fwd_npu_simd,case.params['chunk_size'])
 binding=bindings()
 files=[P/'additional.py',P/'api_only_candidate.py',*[pathlib.Path(v) for v in binding['original_triton_modules'].values()]]
 files+=list((ROOT/'sgl_kernel_npu').rglob('*.py'))
 return fn,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
