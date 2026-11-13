"""Only the 28 frozen torch cases; CPU preparation and guarded NPU execution."""
import collections
import dataclasses
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import traceback

P=Path(__file__).resolve().parent
mode,task_name,out_name=sys.argv[1:]
task=json.loads(Path(task_name).read_text());out=Path(out_name);out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P),str(P/'profile_helpers/scripts/ci/nightly_support')]
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'


def put(name,data):
    f=out/name;q=f.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2,default=str));q.replace(f)


def sha(f):
    with Path(f).open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()


def proof():
    manifest=P/'SOURCES.json';d=json.loads(manifest.read_text())
    assert all(sha(P/n)==h for n,h in d.items()),'frozen task sources changed'
    return sha(manifest)


if mode in ('prepare','cpu-test'):
    class NoNPU(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('torch_npu','acl'):
                raise ModuleNotFoundError('CPU phase forbids NPU import',name=fullname)
    sys.meta_path.insert(0,NoNPU())
elif mode=='device':
    import fcntl
    with open('/run/lock/ascend-npu/device-0.lock') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('NPU worker requires host guard')
else:raise ValueError(mode)

started=time.time()
try:
    source_sha=proof()
    import torch
    torch.set_num_threads(8)
    from benchmarks.api import cases
    from workloads.device import set_run_device
    from workloads.numerics import compare_outputs,describe
    from tools import kernel_task_profile_full32 as full32
    sys.modules['tools.kernel_task_profile']=full32
    from tools import measure_baselines as core
    core.ROOT=P;core.TILEOPS_ROOT=P/'source'
    set_run_device('cpu');torch.manual_seed(task['seed'])
    assert task['key'][0]=='torch-eager' and task['key'][1] in ('SharedExpertMLPFwdOp','FusedMoESharedExpertFwdOp')
    case=next(c for c in cases(task['key'][1]) if c.id==task['key'][2])
    assert core._jsonable(case.params)==core._jsonable(task['case']['params'])

    def transfer(x,device):
        if isinstance(x,torch.Tensor):return x.to(device)
        if isinstance(x,tuple):return tuple(transfer(v,device) for v in x)
        if isinstance(x,list):return [transfer(v,device) for v in x]
        return x

    def digest(xs):
        h=hashlib.sha256()
        def visit(x):
            if isinstance(x,torch.Tensor):
                h.update(str((tuple(x.shape),str(x.dtype),tuple(x.stride()),x.storage_offset())).encode())
                raw=x.detach().contiguous().view(torch.uint8).reshape(-1)
                for part in raw.split(64*1024**2):h.update(part.cpu().numpy().tobytes())
            elif isinstance(x,(tuple,list)):
                for v in x:visit(v)
            else:h.update(repr(x).encode())
        visit(xs);return h.hexdigest()

    if mode=='cpu-test':
        from workloads.moe import ref_shared_expert,gated_activation
        checks=[]
        for dt in (torch.float16,torch.bfloat16):
            torch.manual_seed(98)
            x=torch.randn(3,16,dtype=dt);w=torch.randn(32,16,dtype=dt);d=torch.randn(16,16,dtype=dt)
            act=gated_activation(x.float()@w.float().T,'silu_and_mul')
            required=(act.to(dt).float()@d.float().T).to(dt)
            old=(act@d.float().T).to(dt)
            got=ref_shared_expert(x,w,d)
            assert torch.equal(required,got)
            assert not torch.equal(old,got),'regression input must distinguish missing cast'
            checks.append({'dtype':str(dt),'required_equal':True,'old_differs':True})
        assert not any(n=='torch_npu' or n.startswith('torch_npu.') for n in sys.modules)
        put('CPU_TEST.json',{'passed':checks,'no_device_imports':True,'source_sha256':source_sha})
    elif mode=='prepare':
        xs=case.inputs
        before=digest(xs)
        with torch.no_grad():expected=case.reference(*xs)
        assert digest(xs)==before
        policy=case.verification
        check=compare_outputs(expected,expected,policy)
        controls=[]
        for control in policy.controls:
            with torch.no_grad():bad=control.run(case.reference,xs)
            try:compare_outputs(bad,expected,policy)
            except AssertionError:controls.append({'name':control.name,'rejected':True})
            else:raise AssertionError('negative control accepted: '+control.name)
            del bad
        assert check.checked_outputs==check.total_outputs>0 and not check.unchecked_reason
        assert digest(xs)==before
        from handoff import save
        payload=save({'inputs':xs,'expected':expected},P,task['id'],timeout=600)
        put('payload.json',payload)
        put('prepare.json',{'key':task['key'],'params':core._jsonable(case.params),'seed':task['seed'],
                            'source_sha256':source_sha,'input_sha256':before,'controls':controls,
                            'oracle':'corrected canonical formula independently executed on CPU',
                            'verification':describe(policy),'seconds':time.time()-started})
        assert not any(n=='torch_npu' or n.startswith('torch_npu.') for n in sys.modules)
        put('PREPARED.json',{'source_sha256':source_sha,'seconds':time.time()-started})
    else:
        import torch_npu
        torch.npu.set_device(0)
        torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
        torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        props=torch.npu.get_device_properties(0)
        assert torch.npu.get_device_name(0)=='Ascend950DT_9582'
        assert (props.cube_core_num,props.vector_core_num)==(32,64)
        prep_source=json.loads((out/'prepare.json').read_text())['source_sha256']
        if prep_source!=source_sha:
            compat=json.loads((P/'PREPARATION_COMPATIBILITY.json').read_text())
            assert prep_source==compat['source_sha256']
            old_manifest=P/compat['manifest']
            assert sha(old_manifest)==prep_source
            assert all(sha(P/n)==h for n,h in json.loads(old_manifest.read_text()).items() if n not in ('worker.py','controller.py'))
        from handoff import load
        saved=load(json.loads((out/'payload.json').read_text()))
        cpu=saved['inputs'];expected=saved['expected'];case.__dict__['inputs']=cpu
        policy=case.verification
        args=transfer(cpu,'npu');torch.npu.synchronize()
        before=digest(args)
        assert before==json.loads((out/'prepare.json').read_text())['input_sha256']
        candidate=case.reference
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves

        class NoCPUFallback(TorchDispatchMode):
            def __init__(self):super().__init__();self.ops=collections.Counter()
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {}
                for t in tree_leaves((args,kwargs)):
                    if isinstance(t,torch.Tensor):assert t.device.type=='npu',('CPU tensor in candidate',str(func),str(t.device))
                result=func(*args,**kwargs)
                for t in tree_leaves(result):
                    if isinstance(t,torch.Tensor):assert t.device.type=='npu',('CPU output in candidate',str(func))
                self.ops[str(func)]+=1
                return result

        with torch.no_grad(),NoCPUFallback() as route:
            produced=candidate(*args)
        torch.npu.synchronize()
        check=compare_outputs(transfer(produced,'cpu'),expected,policy)
        assert check.checked_outputs==check.total_outputs>0 and not check.unchecked_reason
        put('device-route.json',{'no_CPU_tensor_arithmetic_or_transfer_in_candidate':True,'dispatch_ops':dict(route.ops),
                                'output_devices':[str(t.device) for t in tree_leaves(produced) if isinstance(t,torch.Tensor)],
                                'hardware':{'model':torch.npu.get_device_name(0),'cube':props.cube_core_num,'vector':props.vector_core_num},
                                'torch':torch.__version__,'torch_npu':torch_npu.__version__})
        del produced
        # Reuse qualified session/isolated timers, while retaining final timed outputs.
        from complete_profile import make_timer,replay_coverage
        from session_profile import UnsupportedSessionError
        final=[]
        def measured(backend):
            calls=0;final.clear()
            def capture(*a):
                nonlocal calls
                value=candidate(*a);calls+=1
                if calls==19:final.append(value)
                return value
            timer=make_timer(out/backend,mode=backend)
            with torch.no_grad():m=timer(torch,torch_npu,capture,args,3,15)
            assert calls==19 and len(final)==1
            return m
        measurement=measured('isolated')
        assert measurement['measurement_protocol']=='ascend950-isolated-ffts-v3'
        measurement.update(warmup=3,repeats=15,flush_bytes=1<<30)
        check=compare_outputs(transfer(final.pop(),'cpu'),expected,policy)
        assert check.checked_outputs==check.total_outputs>0 and not check.unchecked_reason
        assert digest(args)==before,'full input digest changed'
        assert len(replay_coverage(measurement))==15
        assert proof()==source_sha
        put('correctness.json',{'status':'verified','check':dataclasses.asdict(check),'correctness_source':'final-timed-output',
                                'negative_controls':json.loads((out/'prepare.json').read_text())['controls'],
                                'full_inputs_preserved':True,'no_cpu_fallback_route':True})
        put('measurement.json',{'key':task['key'],'source_sha256':source_sha,'input_sha256':before,
                                'preparation_source_sha256':prep_source,'timing':measurement,'hardware':json.loads((out/'device-route.json').read_text())['hardware'],
                                'correctness_sha256':sha(out/'correctness.json'),'metric':'device_latency_ms','seconds':time.time()-started})
        torch.npu.synchronize()
except BaseException as e:
    put(mode+'-failure.json',{'error':str(e),'traceback':traceback.format_exc(),'seconds':time.time()-started})
    raise
