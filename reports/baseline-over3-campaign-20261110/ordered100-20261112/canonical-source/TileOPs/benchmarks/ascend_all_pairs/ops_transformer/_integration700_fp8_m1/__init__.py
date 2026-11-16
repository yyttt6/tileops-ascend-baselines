"""One exact server1 FP8 M1 route; CPU construction, guarded original V5 execution."""
import ast
import builtins
import dataclasses
import hashlib
import json
import os
import platform
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOYMENT = Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_dense48_r2')
RUN_ROOT = DEPLOYMENT.parent / 'threehost_fp8_m1_registration1' / 'runs'
KEY = ('ops-transformer', 'GemmFP8FwdOp', 'ds-v3-m1-down-blk-float8_e4m3fn-bfloat16')
GUARD_OWNER = 'baseline700/threehost_fp8_m1_registration1'
PINS_HASH = '5ae6322586e56146b2f81633351f8a0b4364317960f4eeee21595ff935906060'

def _checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('Frozen dependency changed: ' + str(path))
    return raw

def _metadata():
    pins = json.loads(_checked(ROOT/'LOCAL_PINS.json', PINS_HASH))
    data = {n: _checked(ROOT/n, h) for n,h in pins.items()}
    return pins, data

def _canonical(value):
    from collections.abc import Mapping
    def plain(v):
        if isinstance(v,Mapping): return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)): return [plain(x) for x in v]
        return v
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)

def _key(case, baseline='ops-transformer'):
    return baseline, case._call.signature.name, case.id

def _contract(case):
    c=case._call
    return dict(key=list(_key(case)),params=case.params,call=dict(
        indices=c.indices,rawparams=c.params,tensors=c.tensors,
        specs={n:None if s is None else dataclasses.asdict(s) for n,s in c.specs.items()},
        presence={n:c.present(n) for n in set(c.specs)|{'out'}},has_out=c.has_out,
        signature=dataclasses.asdict(c.signature)))

def matches(case, baseline='ops-transformer'):
    _,data=_metadata()
    try:
        return (_key(case,baseline)==KEY and case._call.case_id==case.id
                and _canonical(_contract(case))==_canonical(json.loads(data['EXACT_CONTRACT.json'])))
    except (AttributeError,TypeError,ValueError,KeyError):
        return False

def _unsupported(message):
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase(message)

def _inputs(args, kwargs):
    if kwargs or len(args)!=4: _unsupported('Exactly (a,b,scale_a,scale_b); bias absent')
    specs=(((1,2048),'float8_e4m3fn'),((7168,2048),'float8_e4m3fn'),
           ((1,16),'float32'),((7168,16),'float32'))
    devices=[]
    for t,(shape,dtype) in zip(args,specs):
        if (tuple(getattr(t,'shape',()))!=shape or str(getattr(t,'dtype','')).removeprefix('torch.')!=dtype
            or str(getattr(t,'layout',''))!='torch.strided'
            or getattr(getattr(t,'device',None),'type',None)!='npu'):
            _unsupported('Exact shape/dtype/strided NPU tensors required')
        devices.append(str(t.device))
    if set(devices)!={'npu:0'}: _unsupported('Inputs must share guarded NPU device 0')

def _host():
    if platform.machine()!='aarch64' or platform.node()!='a5d0b50dc68f' or not DEPLOYMENT.is_dir():
        raise RuntimeError('Wrong deployment: server1 ARM a5d0b50dc68f required; CPU validation is not execution')

def _proc(pid):
    s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return int(s[19]),int(s[2])

def _guard():
    import fcntl
    with Path('/run/lock/ascend-npu/device-0.lock').open() as f:
        m=json.load(f);child=m['child']
        if (m.get('owner')!=GUARD_OWNER or m.get('phase')!='running' or m.get('device')!=0
            or child['pid']!=os.getpgrp()
            or m['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or _proc(child['pid'])!=(child['start_ticks'],child['pid'])
            or _proc(m['pid'])[0]!=m['start_ticks']):
            raise RuntimeError('Live matching foreground npu-guard required')
        try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(f.fileno(),fcntl.LOCK_UN)
            raise RuntimeError('NPU host lock not held')
    return (m['boot_id'],m['pid'],m['start_ticks'],child['pid'],child['start_ticks'])

def _environment():
    for name in ('ASCEND_CUSTOM_OPP_PATH','ASCEND_CLASSIC_TRANSFORMER_VENDOR'):
        if os.environ.get(name)!=str(DEPLOYMENT/'vendor'):
            raise RuntimeError('Original isolated vendor environment required: '+name)
    for name in ('ASCEND_HOME_PATH','ASCEND_CANN_PACKAGE_PATH'):
        if os.environ.get(name)!='/usr/local/Ascend/cann-9.1.0':
            raise RuntimeError('Original CANN 9.1 environment required: '+name)

def _runtime_pins(data):
    pins=json.loads(data['RUNTIME_PINS.json'])
    for group in pins.values():
        for name,digest in group.items():
            h=hashlib.sha256()
            with open(name,'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=digest:raise RuntimeError('Frozen runtime changed: '+name)
    # Additional runtime vendor files could change dispatch; reject the inventory drift.
    expected=set(pins['vendor'])
    actual={str(p) for p in (DEPLOYMENT/'vendor').rglob('*') if p.is_file() and '/ascendc/' not in str(p)}
    if actual!=expected:raise RuntimeError('Vendor runtime inventory changed')
    return pins

def _mapped_identity(stat_inode, stat_device, mapped_inode, mapped_major, mapped_minor):
    # This fixed server1 container exposes overlay st_dev=0:61, while /proc/maps
    # identifies the backing XFS=253:3. Confirmed by same-file readonly mmap.
    pair=(os.major(stat_device),os.minor(stat_device))
    mapped=(mapped_major,mapped_minor)
    return stat_inode==mapped_inode and (pair==mapped or (pair==(0,61) and mapped==(253,3)))

def _maps(pins, require=False):
    expected=pins['loaded']; names={Path(p).name for p in expected};seen=set()
    canonical={str(Path(p).resolve()):h for p,h in expected.items()}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6:continue
        path=fields[5]
        if Path(path.removesuffix(' (deleted)')).name not in names:continue
        if path.endswith(' (deleted)'):raise RuntimeError('Deleted native mapping')
        resolved=str(Path(path).resolve())
        # Some reviewed aliases resolve to one canonical ELF.
        if resolved not in canonical:raise RuntimeError('Unexpected native owner: '+path)
        st=Path(resolved).stat();major,minor=(int(v,16) for v in fields[3].split(':'))
        if not _mapped_identity(st.st_ino, st.st_dev, int(fields[4]), major, minor):
            raise RuntimeError('Native mapped inode/device mismatch: '+path)
        if '/vendor/' in resolved or Path(resolved).name=='libnnopbase.so':
            _checked(resolved,canonical[resolved])
        seen.add(resolved)
    required={str((DEPLOYMENT/'vendor/op_api/lib/libcust_opapi.so').resolve()),
              '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libnnopbase.so'}
    if require and not required<=seen:raise RuntimeError('Missing original V5/native base maps')

def _load(data, directory, pins):
    """Private verified modules: no ambient namesake can replace the math closure."""
    private={};original=builtins.__import__
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level==0 and name in private:return private[name]
        return original(name,globals,locals,fromlist,level)
    names=(('benchmarks.ascend_baselines','ascend_baselines.py'),
           ('benchmarks.ascend_math_more','ascend_math_more.py'),
           ('benchmarks.api','api.py'),('binding','binding.py'))
    for name,filename in names:
        mod=types.ModuleType('_fp8_verified_'+name.replace('.','_'))
        mod.__file__=str(directory/'binding.py') if name=='binding' else str(DEPLOYMENT/'TileOPs/benchmarks'/filename)
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        raw=data['frozen/'+filename]
        if name=='benchmarks.api':
            # Passive dataclass only; avoid unrelated timing/checker imports.
            tree=ast.parse(raw)
            nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Implementation']
            raw=ast.Module(body=ast.parse('import dataclasses\nfrom typing import Callable, Optional').body+nodes,type_ignores=[])
        exec(compile(raw,mod.__file__,'exec'),mod.__dict__)
        private[name]=mod
    # Provenance-only adapter: original binding enumerates all vendor sources. Use
    # reviewed necessary runtime pins, avoiding rereading 2791 historical files.
    # _MathCall, _owned_native, build/run/cleanup bodies remain byte-for-byte original.
    private['benchmarks.ascend_baselines'].source_hashes=lambda *paths:dict(pins['vendor'],**pins['source'])
    return private['binding']

def _build_child(data,pins):
    RUN_ROOT.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='call-',dir=RUN_ROOT))
    (directory/'binding.py').write_bytes(data['frozen/binding.py'])
    (directory/'vendor').symlink_to(DEPLOYMENT/'vendor',target_is_directory=True)
    (directory/'guard.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    binding=_load(data,directory,pins)
    # Original build reads case_id only; never access live case inputs or parameters.
    return binding.build(types.SimpleNamespace(case_id=KEY[2]))

def make(case, baseline='ops-transformer'):
    from benchmarks.ascend_baselines import Candidate
    if not matches(case,baseline):_unsupported('Exact FP8 M1 manifest Call/parameters/tensor declarations required')
    pins,data=_metadata()
    child=None;identity=None;closed=False;runtime=None
    def cleanup():
        nonlocal child,closed
        if child is not None:
            _host()
            if _guard()!=identity:raise RuntimeError('Cleanup requires original guard lifetime')
            child.cleanup()
            child=None
        closed=True
    def run(*args,**kwargs):
        nonlocal child,identity,runtime
        if closed:raise RuntimeError('Candidate already cleaned up')
        _inputs(args,kwargs)
        _host();current=_guard();_environment()
        if identity is not None and identity!=current:raise RuntimeError('Candidate cannot cross guard lifetimes')
        _,fresh=_metadata()
        if child is None:
            runtime=_runtime_pins(fresh)
            _maps(runtime)
            identity=current
            child=_build_child(fresh,runtime)
        try:
            _maps(runtime)
            result=child.implementation.run(*args)
            _maps(runtime,require=True)
            if (tuple(result.shape)!=(1,7168) or str(result.dtype)!='torch.bfloat16'
                or str(result.device)!='npu:0'):
                raise RuntimeError('Original API output violates exact contract')
            return result
        except BaseException:
            cleanup()
            raise
    impl=types.SimpleNamespace(run=run,args=None,reset=None,noncomparable_reason=None)
    return Candidate(impl,'ops-transformer::aclnnGroupedMatmulV5',
        'Original complete V5; operand exchange, scale transpose, output transpose only.',
        {str(ROOT/n):h for n,h in pins.items()},
        limitations=('Exact server1 ARM/CANN9.1 deployment; guarded runtime required.',),cleanup=cleanup)

def wrap(previous):
    def dispatch(case):
        if _key(case)==KEY:return make(case)
        if previous is not None:return previous(case)
        _unsupported('No matching FP8 M1 route')
    dispatch.previous=previous
    return dispatch
