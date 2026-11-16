"""One reviewed exact Bmm contract; original ARM9.1 factories remain runtime-lazy."""
import ast
import builtins
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
import types
from pathlib import Path

LOCAL_ROOT = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_bmm_variant_registration1')
REMOTE_ROOT = Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_variant_registration1')
# Explicit supported deployment roots, independent of the published module location.
DATA_ROOT = REMOTE_ROOT if platform.machine() == 'aarch64' else LOCAL_ROOT
WORKSPACE = '/opt/tileops-all-pairs/workspace'
REPORT = WORKSPACE + '/reports/baseline-integration-700-20261026'
ORIGINAL = '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_backend_full1_r2'
ADAPTER = ORIGINAL + '/adapter.py'
BINDING = ORIGINAL + '/binding.py'
BASE = ORIGINAL + '/TileOPs/benchmarks/ascend_baselines.py'
MATH = ORIGINAL + '/TileOPs/benchmarks/ascend_math_more.py'
API = ORIGINAL + '/TileOPs/benchmarks/api.py'
DATA_HASHES = {'CONTRACTS.json': 'ccc0d1ef37f35e090af334f59e6954dddadc5a0bb1c2dd18baa8032e2a63fbd2', 'PINS.json': 'bb2b7fbfb1eb4b5599a3435497122fe298f8dd62dc3fc5fbae8d07bc7979b35f', 'FROZEN.json': '65ad35cd466df14feb2cc229606fd51c1010a0db62191db25cfc7f56dcf5ed91', 'ORIGINAL_SOURCEHASHES.json': '4d4b94f53f733531fa85b7a9ac71dc89a0e5d2aa40a4026be7af4818e7fcca02'}


def checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('Frozen dependency changed: ' + str(path))
    return raw


def metadata():
    return {n: json.loads(checked(DATA_ROOT/n, h)) for n, h in DATA_HASHES.items()}


def plain(value):
    from collections.abc import Mapping
    if hasattr(value, 'kind') and hasattr(value, 'fields'):
        return {value.kind: plain(value.fields)}
    if isinstance(value, Mapping): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def contract(case):
    c = case._call
    return plain(dict(op=c.signature.name, case_id=case.id, call_case_id=c.case_id,
        parameters=case.params, indices=c.indices, rawparams=c.params, tensors=c.tensors,
        inputs=list(c.signature.inputs), outputs=list(c.signature.outputs),
        specs={n: None if s is None else dict(shape=s.shape,dtype=s.dtype,values=s.values,cpu=s.cpu)
               for n,s in c.specs.items()},
        presence={n:c.present(n) for n in set(c.specs)|set(c.signature.inputs)|{'out'}},has_out=c.has_out))


def key(case): return (case._call.signature.name, case.id)


def rows(): return metadata()['CONTRACTS.json']


def matches(case):
    try:
        return any(key(case)==(r['op'],r['case_id']) and canonical(contract(case))==canonical(r) for r in rows())
    except (AttributeError, KeyError, TypeError, ValueError): return False


def snapshot(runtime=False):
    pins=metadata()['PINS.json'];data={}
    for original, item in pins['sources'].items():
        data[original]=checked(DATA_ROOT/item['file'],item['sha256'])
    if runtime:
        for path,digest in pins['runtime'].items(): checked(path,digest)
    return data


def _implementation(run, data):
    # Exact passive original API class; importing full benchmarks.api imports torch.
    from typing import Callable, Optional
    import dataclasses
    tree=ast.parse(data[API],API)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Implementation')
    ns=dict(dataclasses=dataclasses,Callable=Callable,Optional=Optional)
    exec(compile(ast.Module(body=[node],type_ignores=[]),API,'exec'),ns)
    return ns['Implementation'](run)


def private_load(data):
    """Original complete bytes, private imports and pinned dynamic importlib loaders."""
    private={}; original_import=builtins.__import__
    names={'benchmarks.ascend_baselines':BASE,'benchmarks.ascend_math_more':MATH,'benchmarks.api':API}
    from benchmarks.ascend_baselines import UnsupportedCase
    def load(path):
        path=str(Path(path))
        if path not in data: raise RuntimeError('Unpinned private import: '+path)
        if path in private:return private[path]
        mod=types.ModuleType('_bmm_variant_'+hashlib.sha256(path.encode()).hexdigest()[:16]);mod.__file__=path
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        private[path]=mod
        if path==API:
            mod.Implementation=type(_implementation(lambda:None,data))
        else:exec(compile(data[path],path,'exec'),mod.__dict__)
        if path==BASE:mod.UnsupportedCase=UnsupportedCase
        return mod
    class Loader:
        def __init__(self,path):self.path=path
        def exec_module(self,mod):mod.__dict__.update(load(self.path).__dict__)
    def spec(name,path):
        if str(path) not in data:raise RuntimeError('Unpinned importlib path: '+str(path))
        return types.SimpleNamespace(name=name,loader=Loader(str(path)))
    util=types.SimpleNamespace(spec_from_file_location=spec,module_from_spec=lambda s:types.ModuleType(s.name))
    import importlib
    bridge=types.SimpleNamespace(util=util,import_module=importlib.import_module)
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level==0 and name in names:return load(names[name])
        if name in ('importlib','importlib.util'):return bridge
        return original_import(name,globals,locals,fromlist,level)
    return load


def _proc(pid):
    s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return int(s[19]),int(s[2])


def require_guard():
    import fcntl
    try:
        with Path('/run/lock/ascend-npu/device-0.lock').open('r') as f:
            m=json.load(f);child=m['child'];owner=os.environ.get('BMM_VARIANT_GUARD_OWNER')
            assert owner and m['owner']==owner and m['phase']=='running' and m['device']==0
            assert m['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            assert child['pid']==os.getpgrp()
            assert _proc(child['pid'])==(child['start_ticks'],child['pid'])
            assert _proc(m['pid'])[0]==m['start_ticks']
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)
                raise RuntimeError('Guard kernel lock is not held')
        return m['boot_id'],m['pid'],m['start_ticks'],child['pid'],child['start_ticks'],owner
    except (OSError,AssertionError,KeyError,ValueError,TypeError) as e:
        raise RuntimeError('Live foreground npu-guard required') from e


CACHE_NAMES=('ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR',
 'ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR')


def environment():
    pins=metadata()['PINS.json']
    if platform.machine()!=pins['host']['machine'] or platform.node()!=pins['host']['node']:
        raise RuntimeError('Only tested server1 ARM host is supported')
    if DATA_ROOT!=REMOTE_ROOT or sys.prefix!='/opt/tileops-ci/runtime':
        raise RuntimeError('Tested deployment/runtime required')
    for name,value in pins['environment'].items():
        if os.environ.get(name)!=value:raise RuntimeError('Tested environment required: '+name)
    for name in ('LD_PRELOAD','PYTHONHOME'):
        if os.environ.get(name):raise RuntimeError('Unreviewed loader override: '+name)
    for name in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name) not in (None,'','0'):raise RuntimeError('Physical device 0 required')
    root=Path(os.environ.get('BMM_VARIANT_RUN_ROOT','/invalid')).resolve()
    if root.parent!=REMOTE_ROOT/'runs' or not (root/'RUN_EXCLUSIVE').is_file():
        raise RuntimeError('Fresh private recipe run directory required')
    for name in CACHE_NAMES:
        if Path(os.environ.get(name,'/invalid')).resolve()!=root/name:
            raise RuntimeError('Private cache/log directory required: '+name)
    return tuple(sorted(os.environ.items())),str(root)


def admit(data, runtime=True):
    environment_token=environment() # wrong host fails before imports or native access
    guard=require_guard()
    if runtime:
        for path,digest in metadata()['PINS.json']['runtime'].items(): checked(path,digest)
    loaded_libraries()
    if require_guard()!=guard or environment()!=environment_token:raise RuntimeError('Admission changed')
    return data,guard,environment_token


def validate_inputs(case,args,kwargs):
    from benchmarks.ascend_baselines import UnsupportedCase
    names=list(case._call.signature.inputs)
    if kwargs or len(args)!=len(names):raise UnsupportedCase('Exact positional input arity required')
    for name,value in zip(names,args):
        spec=case._call.specs.get(name)
        if spec is None:
            if value is not None:raise UnsupportedCase('Absent optional input: '+name)
            continue
        if tuple(getattr(value,'shape',()))!=tuple(spec.shape) or str(getattr(value,'dtype','')).removeprefix('torch.')!=spec.dtype:
            raise UnsupportedCase('Exact input shape/dtype required: '+name)
        device=str(getattr(value,'device',''))
        if device!=('cpu' if spec.cpu else 'npu:0'):raise UnsupportedCase('Exact input device required: '+name)


def loaded_libraries():
    pins=metadata()['PINS.json']['runtime']
    expected={str(Path(p).resolve()):h for p,h in pins.items() if '.so' in p}
    names={Path(p).name for p in expected}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or Path(fields[5].removesuffix(' (deleted)')).name not in names:continue
        path=fields[5]
        if path.endswith(' (deleted)') or str(Path(path).resolve()) not in expected:
            raise RuntimeError('Unexpected mapped dependency: '+path)
        if Path(path).stat().st_ino!=int(fields[4]):raise RuntimeError('Mapped dependency replaced: '+path)


def require_backend():
    # Validation only: candidate never changes global backend or oracle.
    import torch
    if torch.backends.mkldnn.enabled is not False or torch.get_num_threads()!=4:
        raise RuntimeError('Approved evaluation requires public mkldnn.enabled=False and 4 CPU threads')
    if str(torch.__version__)!='2.10.0+cpu':
        raise RuntimeError('Tested CPU reference runtime required')


def make_candidate(case):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case):raise UnsupportedCase('Exact Bmm backend variant manifest contract required')
    expected=canonical(contract(case));frozen=copy.deepcopy(case);data=snapshot()
    state={'inner':None,'closed':False,'guard':None,'environment':None}
    def cleanup():
        inner=state['inner']
        if inner is not None:
            if require_guard()!=state['guard'] or environment()!=state['environment']:
                raise RuntimeError('Cleanup requires original live guard and environment')
        state['inner']=None;state['closed']=True
        if inner is not None and inner.cleanup is not None:inner.cleanup()
    def run(*args,**kwargs):
        if state['closed']:raise RuntimeError('Candidate is closed')
        try:
            if canonical(contract(case))!=expected or not matches(case):raise UnsupportedCase('Case changed after construction')
            validate_inputs(frozen,args,kwargs)
            if state['inner'] is not None:
                if require_guard()!=state['guard'] or environment()!=state['environment']:
                    raise RuntimeError('Candidate cannot cross guard/environment lifetimes')
            admitted,guard,env=admit(data, runtime=state['inner'] is None)
            require_backend()
            if state['inner'] is None:
                state['guard']=guard;state['environment']=env
                load=private_load(data)
                mod=load(ADAPTER)
                state['inner']=mod.candidate(frozen,native_factory=load(BINDING).dense_fp32)
            if require_guard()!=guard or environment()!=env:raise RuntimeError('Execution identity changed')
            loaded_libraries()
            result=state['inner'].implementation.run(*args,**kwargs)
            loaded_libraries()
            return result
        except BaseException as error:
            try:cleanup()
            except BaseException as clean_error:raise clean_error from error
            raise
    return Candidate(_implementation(run,data),'ops-transformer::aclnnGroupedMatmulV5',
        'Exact Bmm BF16 contract; requires explicitly approved mkldnn.enabled=False evaluation context; original V5 closure.',
        {**metadata()['ORIGINAL_SOURCEHASHES.json'], **{p:item['sha256'] for p,item in metadata()['PINS.json']['sources'].items()}},cleanup=cleanup)


make=make_candidate

def wrap(previous):
    def factory(case):
        from benchmarks.ascend_baselines import UnsupportedCase
        if key(case) in {(r['op'],r['case_id']) for r in rows()}:return make_candidate(case)
        if previous is None:raise UnsupportedCase('No previous provider for unknown key')
        return previous(case)
    factory.previous=previous
    return factory


def install(providers):
    """Parent invokes against the actual provider dictionary when publishing."""
    for op in {r['op'] for r in rows()}:providers[op]=wrap(providers.get(op))
