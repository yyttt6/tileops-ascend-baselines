"""Record an input-layout diagnostic, then run unchanged full manifest checks."""
from pathlib import Path
import sys, os, json, runpy, hashlib
WS = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(WS/'TileOPs'), str(WS/'TileOPs/src'), str(WS)]
os.environ['TILELANG_CACHE_DIR'] = str(WS/'.cache/all-pairs/tilekernels/tilelang')
os.environ['TMPDIR'] = str(WS/'.cache/all-pairs/tilekernels/tmp')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from scripts.ascend_classic_check import require_guard
require_guard()
import torch, torch_npu
from benchmarks.api import cases
from tools.baseline_protocol import seed_case
from workloads.device import set_run_device
from types import SimpleNamespace
from benchmarks.ascend_all_pairs.tilekernels import pooling_max
set_run_device('npu')
seed = seed_case(torch, 'AdaptiveMaxPool2dFwdOp', 'spp-6x6-float16')
case = next(c for c in cases('AdaptiveMaxPool2dFwdOp') if c.id == 'spp-6x6-float16')
x = case.inputs[0]
input_cpu = x.cpu()
captured = []
original_native = pooling_max.native
def capture(values, k):
    captured.append(values.cpu())
    return torch.zeros((values.shape[0],k), device=values.device,dtype=torch.int64)
try:
    pooling_max.native = lambda *_: capture
    # This is layout inspection only. The placeholder indices are never verified.
    pooling_max.max_pool(case).implementation.run(x)
    cpu_case = SimpleNamespace(_call=case._call, params=case.params, inputs=(input_cpu,))
    pooling_max.max_pool(cpu_case).implementation.run(input_cpu)
finally:
    pooling_max.native = original_native
torch.npu.synchronize()
diagnostic = {'schema':'tilekernels-pooling-input-diagnostic/v1','is_verification_receipt':False,
              'seed':seed,'input_shape':list(x.shape),'packed_shape':list(captured[0].shape),
              'input_finite':bool(torch.isfinite(input_cpu).all()),
              'packed_finite':bool(torch.isfinite(captured[0]).all()),
              'device_pack_equal_cpu_pack':bool(torch.equal(captured[0],captured[1])),
              'input_min':float(input_cpu.min()),'input_max':float(input_cpu.max()),
              'adapter_sha256':hashlib.sha256(Path(pooling_max.__file__).read_bytes()).hexdigest()}
path=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilekernels/pooling_input_diagnostic.json'
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps(diagnostic,indent=2)+'\n')
print('POOLING_INPUT_DIAGNOSTIC',json.dumps(diagnostic),flush=True)
assert diagnostic['input_finite'] and diagnostic['packed_finite'] and diagnostic['device_pack_equal_cpu_pack']
# Establish whether the original reference itself faults before any vendor call.
oracle = {'schema':'tilekernels-original-oracle-diagnostic/v1','is_verification_receipt':False,
          'op':'AdaptiveMaxPool2dFwdOp','case_id':'spp-6x6-float16','seed':seed,
          'candidate_native_calls':0,'reference_device':'npu',
          'reference_source':'TileOPs/workloads/pool.py',
          'reference_source_sha256':hashlib.sha256((WS/'TileOPs/workloads/pool.py').read_bytes()).hexdigest()}
cpu_reference_output = case.reference(input_cpu)
oracle.update(cpu_reference_same_function_passed=True,
              cpu_output_shape=list(cpu_reference_output.shape),
              cpu_output_dtype=str(cpu_reference_output.dtype))
try:
    reference_output = case.reference(*case.inputs)
    torch.npu.synchronize()
    oracle['status'] = 'reference_only_synchronized'
except Exception as exc:
    import traceback
    oracle.update(status='original_reference_device_failure',error_type=type(exc).__name__,
                  error=str(exc),traceback=traceback.format_exc())
finally:
    oracle_path = path.with_name('original_pooling_oracle_diagnostic.json')
    if oracle_path.exists(): raise FileExistsError(oracle_path)
    oracle_path.write_text(json.dumps(oracle,indent=2)+'\n')
    print('ORIGINAL_ORACLE_DIAGNOSTIC',json.dumps(oracle),flush=True)
if oracle['status'] != 'reference_only_synchronized': raise SystemExit(78)
del captured, input_cpu, x, case, cpu_case
runpy.run_path(str(Path(__file__).with_name('batch.py')),run_name='__main__')
