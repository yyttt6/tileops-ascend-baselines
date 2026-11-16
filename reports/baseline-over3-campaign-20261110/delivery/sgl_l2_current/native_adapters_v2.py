import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='VectorNormFwdOp' and case.params['ord']==2 and not fixed
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.fla.layernorm_gated import layer_norm_fwd_npu_simd,layer_norm_fwd_npu
 from sgl_kernel_npu.moe.mul_add import mul_add
 from original_l2 import build,build_reciprocal,build_affine
 from row_layout_original import row_layout
 bound=types.SimpleNamespace(inputs=args,params=case.params)
 _,width,_=row_layout(bound)
 run=build(bound,layer_norm_fwd_npu if width>8192 else layer_norm_fwd_npu_simd,build_reciprocal(layer_norm_fwd_npu),build_affine(mul_add))
 files=[ROOT/'native_adapters_v2.py',ROOT/'original_l2.py',ROOT/'row_layout_original.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
