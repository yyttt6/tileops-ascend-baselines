import sys,pathlib,runpy
P=pathlib.Path(__file__).resolve().parent
# Original vendor launchers already put Torch2.10 first. Template launchers set
# this path explicitly; missing pure-Python deps are appended, never shadow Torch.
if 'ORIGINAL_VENDOR_PROFILE' not in __import__('os').environ:
 sys.path[:0]=['/opt/baseline-over3-20261110/unified-stack-v1/references/fla-v1/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages','/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe']
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
original_tilelang_vendor=__import__('os').environ.get('BASELINE_ORIGINAL_TILELANG_VENDOR')
vendor=original_tilelang_vendor or __import__('os').environ.get('BASELINE_ORIGINAL_TLA_VENDOR')
if vendor:sys.path[:0]=[vendor,str(pathlib.Path(vendor)/'3rdparty/tvm/python'),str(pathlib.Path(vendor)/'3rdparty/tvm/ffi/python')]
if __import__('os').environ.get('BASELINE_ORIGINAL_TLA_VENDOR'):__import__('setuptools')
script={'prepare':'prepare_rollout.py','cpu-registry-preflight':'cpu_tilelang_registry_preflight.py','fla-candidate':'worker_fla_candidate.py','sgl-candidate':'worker_sgl_wrapper_candidate.py','vllm-norm-candidate':'worker_vllm_norm_candidate.py','vllm-norm-expand':'worker_vllm_norm_candidate_v2.py','vllm-source-path':'worker_vllm_source_path_repair.py'}.get(sys.argv[1],'worker_rollout_compact_v3.py')
if original_tilelang_vendor and script=='worker_rollout_compact_v3.py':script='worker_tilelang_original.py'
if __import__('os').environ.get('BASELINE_ORIGINAL_TLA_VENDOR') and script=='worker_rollout_compact_v3.py':script='worker_tla_cache_evidence.py'
sys.argv=[str(P/script),*sys.argv[2:]];runpy.run_path(str(P/script),run_name='__main__')
