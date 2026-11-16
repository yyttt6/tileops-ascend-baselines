"""Nine reviewed exact contracts; original ARM9.1 factories remain runtime-lazy."""
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

LOCAL_ROOT = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_dsa_registration2')
REMOTE_ROOT = Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_dsa_registration2')
# Explicit supported deployment roots, independent of the published module location.
DATA_ROOT = REMOTE_ROOT if platform.machine() == 'aarch64' else LOCAL_ROOT
WORKSPACE = '/opt/tileops-all-pairs/workspace'
REPORT = WORKSPACE + '/reports/baseline-integration-700-20261026'
ORIGINAL = REPORT + '/threehost_attention_moe53'
BASE = ORIGINAL + '/frozen/TileOPs/benchmarks/ascend_baselines.py'
API = ORIGINAL + '/frozen/TileOPs/benchmarks/api.py'
DSA = ORIGINAL + '/dsa.py'
DATA_HASHES = {'CONTRACTS.json': 'e933446f95554e415505b2a32e39a7f0ca1517287a85c3655914301e60e4c792', 'PINS.json': '9d33755a3566d299d246191997a5447964df6c8a80b23f45a58c77d667bbdc26', 'FROZEN.json': '49701152ee5a3af532354d36c4544262a14ad339b9d6c3105db60a78bc8184b3'}


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


def values_pin(values):
    if values is None:return None
    return {'encoding':'canonical-json-sha256','sha256':hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()}


def contract(case):
    c = case._call
    return plain(dict(op=c.signature.name, case_id=case.id, call_case_id=c.case_id,
        parameters=case.params, indices=c.indices, rawparams=c.params, tensors=c.tensors,
        inputs=list(c.signature.inputs), outputs=list(c.signature.outputs),
        specs={n: None if s is None else dict(shape=s.shape,dtype=s.dtype,values=values_pin(s.values),cpu=s.cpu)
               for n,s in c.specs.items()},
        presence={n:c.present(n) for n in set(c.specs)|set(c.signature.inputs)|{'out'}},has_out=c.has_out))


def key(case): return (case._call.signature.name, case.id)


def rows(): return metadata()['CONTRACTS.json']


def matches(case):
    try:
        r=next((r for r in rows() if key(case)==(r['op'],r['case_id'])),None)
        if r is None:return False
        c=case._call
        for name,value in [('parameters',case.params),('indices',c.indices),('rawparams',c.params),('tensors',c.tensors),('inputs',list(c.signature.inputs)),('outputs',list(c.signature.outputs)),('has_out',c.has_out),('call_case_id',c.case_id)]:
            if canonical(value)!=canonical(r[name]):return False
        if set(c.specs)!=set(r['specs']):return False
        for name,spec in c.specs.items():
            expected=r['specs'][name]
            if spec is None:
                if expected is not None:return False
            elif expected is None or any(canonical(getattr(spec,k))!=canonical(expected[k]) for k in ('shape','dtype','cpu')):return False
            elif (spec.values is None)!=(expected['values'] is None):return False
        return canonical(contract(case))==canonical(r)
    except (AttributeError, KeyError, TypeError, ValueError):return False


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
    """Execute unchanged source bytes; private import wiring and hash metadata only."""
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    cfg=metadata()['FROZEN.json']; private={}; original_import=builtins.__import__
    names={'benchmarks.ascend_baselines':BASE,'benchmarks.api':API}
    names.update({'benchmarks.'+n:p for n,p in cfg['sources'].items() if n.startswith('ascend_')})
    def load(path):
        if path in private:return private[path]
        if path not in data:raise RuntimeError('Unpinned source: '+path)
        mod=types.ModuleType('_dsa2_'+hashlib.sha256(path.encode()).hexdigest()[:16])
        mod.__file__=str(DATA_ROOT/metadata()['PINS.json']['sources'][path]['file'])
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        private[path]=mod
        if path==API:mod.Implementation=type(_implementation(lambda:None,data))
        else:exec(compile(data[path],mod.__file__,'exec'),mod.__dict__)
        if path==BASE:mod.Candidate=Candidate;mod.UnsupportedCase=UnsupportedCase
        return mod
    def verified(group):
        if group!='attention':raise RuntimeError('Unsupported source group')
        # admit() verified the entire runtime pin set once for this candidate.
    bridge=types.SimpleNamespace(verify_pins=verified,load=lambda name:load(cfg['sources'][name]))
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level==0 and name=='benchmarks.ascend_all_pairs.ops_transformer.validated_runtime':return bridge
        if level==0 and name in names:return load(names[name])
        return original_import(name,globals,locals,fromlist,level)
    return load


def _proc(pid):
    s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return int(s[19]),int(s[2])


def require_guard():
    import fcntl
    try:
        with Path('/run/lock/ascend-npu/device-0.lock').open('r') as f:
            m=json.load(f);child=m['child'];owner=os.environ.get('DSA2_GUARD_OWNER')
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
        raise RuntimeError('Only tested server2 ARM host is supported')
    if DATA_ROOT!=REMOTE_ROOT or sys.prefix!='/opt/tileops-ci/runtime':
        raise RuntimeError('Tested deployment/runtime required')
    for name,value in pins['environment'].items():
        if os.environ.get(name)!=value:raise RuntimeError('Tested environment required: '+name)
    for name in ('LD_PRELOAD','PYTHONHOME'):
        if os.environ.get(name):raise RuntimeError('Unreviewed loader override: '+name)
    for name in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name) not in (None,'','0'):raise RuntimeError('Physical device 0 required')
    root=Path(os.environ.get('DSA2_RUN_ROOT','/invalid')).resolve()
    if root.parent!=REMOTE_ROOT/'runs' or not (root/'RUN_EXCLUSIVE').is_file():
        raise RuntimeError('Fresh private recipe run directory required')
    for name in CACHE_NAMES:
        if Path(os.environ.get(name,'/invalid')).resolve()!=root/name:
            raise RuntimeError('Private cache/log directory required: '+name)
    import shutil
    cfg=metadata()['FROZEN.json']['groups']['attention']
    if os.environ.get('LD_LIBRARY_PATH')!=cfg['ld_library_path_prefix']:
        raise RuntimeError('Original server2 LD_LIBRARY_PATH required')
    if os.environ.get('ASCEND_CANN_PACKAGE_PATH')!='/usr/local/Ascend/cann-9.1.0':
        raise RuntimeError('Compiler SDK path required')
    for exe,expected in [('bisheng','/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng')]:
        got=shutil.which(exe)
        if got is None or Path(got).resolve()!=Path(expected).resolve():raise RuntimeError('Compiler PATH mismatch: '+exe)
    overlay=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
    for package in ('torch','torch_npu'):
        candidates=[Path(p)/package/'__init__.py' for p in sys.path if (Path(p)/package/'__init__.py').is_file()]
        if not candidates or candidates[0].resolve()!=(overlay/package/'__init__.py').resolve():
            raise RuntimeError('Original overlay must precede base site-packages: '+package)
    if len(os.environ.get('TMPDIR',''))>60:raise RuntimeError('Use short /proc/self/cwd/TMPDIR alias')
    return tuple(sorted(os.environ.items())),str(root)


def admit(runtime=True):
    environment_token=environment() # wrong host fails before imports or native access
    guard=require_guard()
    data=snapshot(runtime=runtime)
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


def synchronize():
    import torch
    torch.npu.synchronize()


def make_candidate(case):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case):raise UnsupportedCase('Exact dsa2 manifest contract required')
    expected=canonical(contract(case))
    # Never copy Case cached inputs/workload: they may already contain device tensors.
    frozen=types.SimpleNamespace(id=case.id,params=copy.deepcopy(case.params),_call=copy.deepcopy(case._call))
    data=snapshot()
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
        if canonical(contract(case))!=expected or not matches(case):raise UnsupportedCase('Case changed after construction')
        try:
            validate_inputs(frozen,args,kwargs)
            if state['inner'] is not None:
                if require_guard()!=state['guard'] or environment()!=state['environment']:
                    raise RuntimeError('Candidate cannot cross guard/environment lifetimes')
            data,guard,env=admit(runtime=state['inner'] is None)
            if state['inner'] is None:
                state['guard']=guard;state['environment']=env
                load=private_load(data)
                mod=load(DSA)
                state['inner']=mod.factory(frozen)
            if require_guard()!=guard or environment()!=env:raise RuntimeError('Execution identity changed')
            loaded_libraries()
            result=state['inner'].implementation.run(*args,**kwargs)
            synchronize()
            loaded_libraries()
            return result
        except BaseException as error:
            try:cleanup()
            except BaseException as clean_error:raise clean_error from error
            raise
    return Candidate(_implementation(run,data),'ops-transformer::FlashAttentionScore',
        'Exact dsa2 contract; lazy original complete ARM9.1 factory; private pinned closure.',
        {p:hashlib.sha256(raw).hexdigest() for p,raw in data.items()},cleanup=cleanup)


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
