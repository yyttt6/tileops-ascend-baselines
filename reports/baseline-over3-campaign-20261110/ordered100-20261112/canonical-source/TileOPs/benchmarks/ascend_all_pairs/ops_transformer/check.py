#!/usr/bin/env python3
"""Reuse the exact classic checker with an isolated provider registry."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path=[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT.parent)]
spec = importlib.util.spec_from_file_location('original_classic_checker', ROOT / 'scripts/ascend_classic_check.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.require_guard()
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from benchmarks.ops import bench_ascend_classics
current_case=None
log_offsets={}
rope_ops={'RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','RoPENeoxPositionIdsFwdOp'}
def factories(case):
    global current_case
    current_case=case
    return {'ops-transformer':lambda:PROVIDERS[case._call.signature.name](case)}
bench_ascend_classics.candidate_factories=factories
original_cpu_reference=checker.cpu_reference
def cpu_reference(torch,reference,inputs):
    if current_case._call.signature.name in rope_ops:
        return reference  # Preserve the original manifest's device-generated tables.
    if current_case._call.signature.name == 'NSAVarlenFwdOp':
        import copy
        # Same bound oracle and values, including its hidden gate tensor. Only
        # the CPU transport changes; the original workload stays untouched.
        owner = copy.copy(reference.__self__)
        owner.g_slc = owner.g_slc.detach().cpu()
        reference = reference.__func__.__get__(owner, type(owner))
    wrapped=original_cpu_reference(torch,reference,inputs)
    if current_case._call.signature.name == 'NSACompressedVarlenFwdOp':
        from benchmarks.ascend_all_pairs.ops_transformer.reference_deps import original_tilelang_math
        def run(*args):
            with original_tilelang_math():
                return wrapped(*args)
        return run
    return wrapped
checker.cpu_reference=cpu_reference
import workloads.numerics as numerics
original_verify=numerics.verify
rope_audit={}
def verify(reference,inputs,**kwargs):
    result=original_verify(reference,inputs,**kwargs)
    if current_case._call.signature.name in rope_ops:
        import torch
        from benchmarks.ascend_all_pairs.ops_transformer.adapters import rope_coefficients
        x,*pos=inputs;position=pos[0] if pos else None
        cs=rope_coefficients(current_case,x,position)
        dim=current_case.params.get('rotary_dim') or x.shape[-1]
        cos,sin=[torch.cat((t.cpu(),t.cpu()),-1).float() for t in cs]
        if x.ndim==4:cos,sin=cos[None,:,None,:],sin[None,:,None,:]
        elif x.ndim==3:cos,sin=cos[:,None,:],sin[:,None,:]
        xc=x.cpu();r=xc[...,:dim].float();half=dim//2
        if current_case.params.get('rope_layout')=='interleaved':
            cos=cos[...,:half].repeat_interleave(2,-1);sin=sin[...,:half].repeat_interleave(2,-1)
            rotated=torch.stack((-r[...,1::2],r[...,0::2]),-1).flatten(-2)
        else:rotated=torch.cat((-r[...,half:],r[...,:half]),-1)
        expected=(r*cos+rotated*sin).to(x.dtype)
        if dim!=x.shape[-1]:expected=torch.cat((expected,xc[...,dim:]),-1)
        request=kwargs['requests']['ops-transformer']
        actual=request.run(*inputs)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(),expected,rtol=1e-3,atol=1e-3)
        rope_audit.update(independent_cpu_rotation_with_device_tables=True,frequency_source='Original manifest frequency tables, same device/dtype')
    return result
numerics.verify=verify
from benchmarks.ascend_graph_call import GraphCall
original_call = GraphCall.call
calls = []
def traced_call(self, op, inputs, outputs, attrs, **kwargs):
    result = original_call(self, op, inputs, outputs, attrs, **kwargs)
    calls.append({'op': op, 'vendor': str(self.vendor), 'api': 'aclopCompileAndExecute',
                  'inputs': [None if x is None else {'shape': list(x.shape), 'dtype': str(x.dtype)} for x in inputs],
                  'attributes': attrs, 'returned': 0})
    return result
GraphCall.call = traced_call

from benchmarks.ascend_all_pairs.ops_transformer.native import GraphCall as OwnGraphCall
own_original_call = OwnGraphCall.call
def own_traced_call(self,op,inputs,outputs,attrs,**kwargs):
    result=own_original_call(self,op,inputs,outputs,attrs,**kwargs)
    calls.append({'op':op,'vendor':str(self.vendor),'api':self.last_api,
                  'inputs':[None if x is None else {'shape':list(x.shape),'dtype':str(x.dtype)} for x in inputs],
                  'attributes':attrs,'returned':0})
    return result
OwnGraphCall.call=own_traced_call

def isolated_compiler(cls):
    original_init=cls.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        import ctypes,os
        option=self.compiler.aclSetCompileopt
        option.argtypes=[ctypes.c_int,ctypes.c_char_p]
        self.check(option(8,os.environ['ASCEND_CACHE_PATH'].encode()),'aclSetCompileopt(CACHE_DIR)')
    cls.__init__=init
isolated_compiler(GraphCall)
isolated_compiler(OwnGraphCall)

def one(arguments):
    calls.clear()
    rope_audit.clear()
    from benchmarks.ascend_all_pairs.ops_transformer.adapters import DISPATCH_EVENTS
    DISPATCH_EVENTS.clear()
    sys.argv=[sys.argv[0],*arguments]
    out=Path(sys.argv[sys.argv.index('--output')+1]);out.parent.mkdir(parents=True,exist_ok=True)
    capture_path=ROOT.parent/'.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/native_log_capture.py'
    spec_capture=importlib.util.spec_from_file_location('original_native_log_capture',capture_path)
    capture=importlib.util.module_from_spec(spec_capture);spec_capture.loader.exec_module(capture)
    import os
    recorder=capture.NativeLogCapture(os.environ['ASCEND_PROCESS_LOG_PATH'],out.with_suffix('.native.jsonl'))
    recorder.state.update(log_offsets)
    recorder.markers += (b'[ERROR]',)
    with recorder:
        rc=checker.main()
    log_offsets.update(recorder.state)
    row=json.loads(out.read_text())
    row['provider_module']='benchmarks.ascend_all_pairs.ops_transformer'
    import os
    row['runtime_pid']=os.getpid()
    row['runtime_log_path']=os.environ.get('ASCEND_PROCESS_LOG_PATH')
    if row['op'] in rope_ops:
        row.update(reference_device='npu',reference_engine='original manifest tables and rotation',**rope_audit)
    if row['op'] == 'NSAVarlenFwdOp':
        row['reference_transport'] = 'Original bound workload copied with g_slc moved to CPU, matching argument transport; no math change.'
    if row['op'] == 'NSACompressedVarlenFwdOp':
        from benchmarks.ascend_all_pairs.ops_transformer.reference_deps import source_evidence
        row['reference_dependency'] = source_evidence()
    row['native_graph_calls']=list(calls)
    row['native_dispatch_calls']=list(DISPATCH_EVENTS)
    row['input_protection']='Original workloads.numerics.Request(preserve_inputs=True)'
    row['source_attribution']={'selected_vendors':sorted({c['vendor'] for c in calls}),
                              'original_native_graph_calls_completed':len(calls),
                              'runtime_synchronized':row.get('status') not in ('device_sync_failed','cleanup_failed')}
    import hashlib
    import inspect
    reference_file=inspect.getsourcefile(type(current_case.workload)) if current_case is not None else None
    if reference_file:
        row['reference_workload_source']={'file':reference_file,'sha256':hashlib.sha256(Path(reference_file).read_bytes()).hexdigest(),'class':type(current_case.workload).__qualname__}
    row['provider_module_sha256']=hashlib.sha256((Path(__file__).with_name('__init__.py')).read_bytes()).hexdigest()
    row['local_checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    row['native_log_capture']=str(out.with_suffix('.native.jsonl'))
    snapshot_root=ROOT.parent/'.cache/all-pairs/ops_transformer/source_snapshots'
    snapshots={}
    for rel,expected in row.get('source_sha256',{}).items():
        if rel.startswith('TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'):
            source=ROOT.parent/rel
            data=source.read_bytes();actual=hashlib.sha256(data).hexdigest()
            if actual!=expected:raise RuntimeError('Adapter changed during the loaded process: '+rel)
            dest=snapshot_root/expected/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(data)
            snapshots[rel]=str(dest)
    row['source_snapshots']=snapshots
    out.write_text(json.dumps(row,indent=2)+'\n')
    return rc

if '--queue' in sys.argv:
    import time,gc
    queue_path=Path(sys.argv[sys.argv.index('--queue')+1]);queue=json.loads(queue_path.read_text())
    start=time.monotonic()
    limit=int(sys.argv[sys.argv.index('--seconds')+1]) if '--seconds' in sys.argv else 800
    for item in queue:
        if isinstance(item, dict):
            required = Path(item['requires_verified_output'])
            if not required.exists() or json.loads(required.read_text()).get('status') != 'verified':
                continue
            args = item['arguments']
        else:
            args = item
        output=Path(args[args.index('--output')+1])
        if output.exists():continue
        if time.monotonic()-start>limit:break
        rc=one(args)
        gc.collect()
        if rc==78:raise SystemExit(78)
    raise SystemExit(0)
raise SystemExit(one(sys.argv[1:]))
