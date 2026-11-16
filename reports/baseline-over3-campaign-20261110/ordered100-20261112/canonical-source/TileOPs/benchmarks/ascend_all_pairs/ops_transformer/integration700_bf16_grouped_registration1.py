"""Exact reviewed BF16 Grouped NT registration; pending parent publication."""
import builtins, hashlib, json, os, types, tempfile, shutil, sys, ast
from pathlib import Path
from types import MappingProxyType
ROOT = '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18'
MANIFEST = '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_grouped_registration1/FROZEN_PINS.json'
MANIFEST_HASH = '7e50a2639b2073d206439dd1f7a68452db1a63473b6242e467c2fe2aa006709e'
LOCAL_FILES = ['/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/resource_gate.py',
 '/root/dyq/workspace/TileOPs/benchmarks/ascend_baselines.py',
 '/root/dyq/workspace/TileOPs/benchmarks/ascend_math_more.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/binding.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_reference_budget2/adapter.py',
 '/root/dyq/workspace/TileOPs/benchmarks/api.py']
CASE_PARAMS = {('ops-transformer', 'GroupedGemmFwdOp', 'moe-fwd-bfloat16'): '{"a":[[4096,4096],"bfloat16"],"b":[[16,4096,4096],"bfloat16"],"batch_offsets":[[16],"int32"],"batch_sizes":[[16],"int32"],"output":[[4096,4096],"bfloat16"],"transpose_a":false,"transpose_b":true}'}
CASE_INDICES = {('ops-transformer', 'GroupedGemmFwdOp', 'moe-fwd-bfloat16'): '{"G":16,"K":4096,"M":4096,"N":4096,"T":"bfloat16","transpose_a":false,"transpose_b":true}'}
CALL_CONTRACTS = {('ops-transformer', 'GroupedGemmFwdOp', 'moe-fwd-bfloat16'): '{"has_out":false,"indices":{"G":16,"K":4096,"M":4096,"N":4096,"T":"bfloat16","transpose_a":false,"transpose_b":true},"presence":{"a":true,"b":true,"batch_offsets":true,"batch_sizes":true,"out":false,"output":true},"rawparams":{"transpose_a":false,"transpose_b":true},"specs":{"a":{"cpu":false,"dtype":"bfloat16","shape":[4096,4096],"values":null},"b":{"cpu":false,"dtype":"bfloat16","shape":[16,4096,4096],"values":null},"batch_offsets":{"cpu":false,"dtype":"int32","shape":[16],"values":[0,256,512,768,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840]},"batch_sizes":{"cpu":false,"dtype":"int32","shape":[16],"values":[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]},"output":{"cpu":false,"dtype":"bfloat16","shape":[4096,4096],"values":null}},"tensors":{"a":[[4096,4096],"bfloat16"],"b":[[16,4096,4096],"bfloat16"],"batch_offsets":[[16],"int32"],"batch_sizes":[[16],"int32"],"output":[[4096,4096],"bfloat16"]}}'}
ENVELOPES = {'moe-fwd-bfloat16': {'cost': 68719476736,
                      'device_required': 11811162112,
                      'disk_required': 3489661440,
                      'elements_including_full_batch_weights_outputs_packing': 301989952,
                      'host_required': 14227081728,
                      'params': {'a': [[4096, 4096], 'bfloat16'],
                                 'b': [[16, 4096, 4096], 'bfloat16'],
                                 'batch_offsets': [[16], 'int32'],
                                 'batch_sizes': [[16], 'int32'],
                                 'output': [[4096, 4096], 'bfloat16'],
                                 'transpose_a': False,
                                 'transpose_b': True}}}
ENV = {'ASCEND_AICPU_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'ASCEND_CLASSIC_TRANSFORMER_VENDOR': '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer',
 'ASCEND_CUSTOM_OPP_PATH': '/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer:/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib:/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer',
 'ASCEND_GLOBAL_LOG_LEVEL': '0',
 'ASCEND_HOME_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'ASCEND_OPP_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2/opp',
 'ASCEND_TOOLKIT_HOME': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'MKL_NUM_THREADS': '4',
 'OMP_NUM_THREADS': '4',
 'PYTHONDONTWRITEBYTECODE': '1'}
LOADED_VENDOR = {'/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so': '179dfb938b1f118558d14e04a189a400d23780c97c6ba25fdbf14dbc897bfef6',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so': '653085d077d6985cd7a6338095e59810a45e7efef0a952e1cbd7f1ca6b040547',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so': '4057480dec2fda046551d2adeed8aaa11b4cc05cbc2154f22ee555ecbda67dd9'}
REQUIRED_LOADED = ('/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libnnopbase.so')

ROOT = Path(ROOT)
def _overlay_mapping_matches(path, digest, fields):
    """Compare kernel identities with a fresh read-only mapping of pinned bytes.

    Overlay pathname stat may expose a different device than /proc/maps. No DSO
    is loaded or executed here. Never accept the device mismatch on inode alone.
    """
    import mmap
    before = {line.split()[0] for line in Path('/proc/self/maps').read_text().splitlines()}
    with open(path, 'rb') as source:
        st = os.fstat(source.fileno())
        if st.st_ino != int(fields[4]) or not st.st_size:
            return False
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            if hashlib.sha256(mapping).hexdigest() != digest:
                return False
            identities = set()
            for line in Path('/proc/self/maps').read_text().splitlines():
                entry = line.split(maxsplit=5)
                if (len(entry) == 6 and entry[0] not in before
                    and entry[5] == path and int(entry[2], 16) == 0):
                    identities.add((entry[3], int(entry[4])))
            after = Path(path).stat()
            stable = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            return stable and (fields[3], int(fields[4])) in identities

def _loaded_vendor_pins(require=False):
    expected = LOADED_VENDOR
    names = {Path(p).name for p in expected}; seen = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6: continue
        path = fields[5]
        if Path(path.removesuffix(' (deleted)')).name not in names: continue
        if path.endswith(' (deleted)'): raise RuntimeError('Deleted vendor mapping')
        resolved = str(Path(path).resolve())
        if resolved not in expected: raise RuntimeError('Unexpected vendor mapping: ' + path)
        st = Path(resolved).stat(); major, minor = (int(x, 16) for x in fields[3].split(':'))
        if st.st_ino != int(fields[4]):
            raise RuntimeError('Mapped vendor inode differs from pinned file: ' + path)
        if (os.major(st.st_dev), os.minor(st.st_dev)) != (major, minor):
            if not _overlay_mapping_matches(resolved, expected[resolved], fields):
                raise RuntimeError('Mapped vendor device identity cannot be proved: ' + path)
        _checked(resolved, expected[resolved]); seen.add(resolved)
    if require and not set(REQUIRED_LOADED).issubset(seen): raise RuntimeError('Missing reviewed loaded vendor libraries')


def _checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('Frozen BF16 GroupedGemm dependency changed: ' + str(path))
    return raw

# The complete manifest is fixed by the published module, never trusted dynamically.
PINS = MappingProxyType(json.loads(_checked(MANIFEST, MANIFEST_HASH)))
CASE_PARAMS = MappingProxyType(CASE_PARAMS)
ENV = MappingProxyType(ENV)
LOADED_VENDOR = MappingProxyType(LOADED_VENDOR)

def _snapshot(runtime=False):
    _checked(MANIFEST, MANIFEST_HASH)
    data = {p: _checked(p, PINS[p]) for p in LOCAL_FILES}
    if runtime:
        for p, digest in PINS.items():
            if p in data: continue
            with open(p, 'rb') as f:
                observed = hashlib.file_digest(f, 'sha256').hexdigest()
            if observed != digest: raise RuntimeError('Frozen runtime/source changed: '+p)
    return MappingProxyType(data)

def _key(case, baseline='ops-transformer'):
    return baseline, case._call.signature.name, case.id

def _plain(v):
    from collections.abc import Mapping
    if isinstance(v, Mapping): return {k:_plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [_plain(x) for x in v]
    return v

def _canonical(value):
    return json.dumps(_plain(value),sort_keys=True,separators=(',', ':'),allow_nan=False)

def _call_contract(call):
    return dict(indices=call.indices,rawparams=call.params,tensors=call.tensors,
        specs={n:None if s is None else dict(shape=s.shape,dtype=s.dtype,values=s.values,cpu=s.cpu) for n,s in call.specs.items()},
        presence={n:call.present(n) for n in set(call.specs)|{'out'}},has_out=call.has_out)

def matches(case,baseline='ops-transformer'):
    try:
        return (case._call.case_id==case.id and _canonical(case.params)==CASE_PARAMS.get(_key(case,baseline))
                and _canonical(_call_contract(case._call))==CALL_CONTRACTS.get(_key(case,baseline)))
    except (AttributeError,KeyError,TypeError,ValueError):return False

def _load(data):
    private={};original=builtins.__import__
    from benchmarks.ascend_baselines import UnsupportedCase
    paths={'resource_gate':LOCAL_FILES[0],'benchmarks.ascend_baselines':LOCAL_FILES[1],
           'benchmarks.ascend_math_more':LOCAL_FILES[2],'binding':LOCAL_FILES[3],
           'adapter':LOCAL_FILES[4],'benchmarks.api':LOCAL_FILES[5]}
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level==0 and name in paths:return private[name]
        return original(name,globals,locals,fromlist,level)
    for name,path in paths.items():
        mod=types.ModuleType('_bf16_grouped_verified_'+name.replace('.','_'));mod.__file__=path
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        code=data[path]
        if name=='benchmarks.api':
            tree=ast.parse(code,path)
            nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
            imports=ast.parse('import dataclasses, functools\nfrom typing import Any, Callable, Optional\nfrom tileops.manifest.workload import Call\nfrom tileops.manifest.values import ADTValue').body
            code=ast.Module(body=imports+nodes,type_ignores=[])
        exec(compile(code,path,'exec'),mod.__dict__)
        if name=='benchmarks.ascend_baselines':mod.UnsupportedCase=UnsupportedCase
        private[name]=mod
    return private

def _freeze(value):
    import dataclasses
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value,**{f.name:_freeze(getattr(value,f.name)) for f in dataclasses.fields(value) if f.init})
    if isinstance(value,dict):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)):return tuple(_freeze(v) for v in value)
    if isinstance(value,set):return frozenset(_freeze(v) for v in value)
    return value

def _frozen_case(key):
    # Build real manifest Call and original passive Case AST, never synthetic contracts.
    from tileops.manifest import load_manifest,load_adts,load_workloads
    from tileops.manifest.plan import entry_plan
    from tileops.manifest.workload import instantiate
    plan=entry_plan(key[1],load_manifest()[key[1]],load_adts(),resolve=False)
    found=[instantiate(plan,row,dt) for row in load_workloads(key[1]) for dt in row.get('dtype_cases') or [{}]]
    call=next(c for c in found if c.case_id==key[2])
    private=_load(_snapshot());OriginalCase=private['benchmarks.api'].Case
    class FrozenCase(OriginalCase):
        def __setattr__(self,name,value):
            if self.__dict__.get('_sealed',False):raise TypeError('Frozen exact case')
            super().__setattr__(name,value)
    frozen=FrozenCase(key[2],_freeze(call),None)
    frozen.__dict__['params']=_freeze(frozen.params)
    frozen._sealed=True
    if not matches(frozen,key[0]):raise RuntimeError('Manifest differs from frozen exact contract')
    return frozen


def _proc(pid):
    s = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return int(s[19]), int(s[2])  # start ticks, process group


def _in_guard_group():
    try:
        metadata=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
        return metadata.get('child',{}).get('pid')==os.getpgrp()
    except (OSError,ValueError,TypeError):
        return False


def _guard_identity():
    """Check running foreground group, live supervisor/child and held kernel lock."""
    import fcntl
    try:
        lock = Path('/run/lock/ascend-npu/device-0.lock')
        with lock.open('r') as f:
            m = json.load(f); child = m['child']; owner = os.environ.get('BF16_GROUPED_GUARD_OWNER')
            if not owner or m['owner'] != owner or m['phase'] != 'running' or m['device'] != 0: return None
            if m['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip(): return None
            if child['pid'] != os.getpgrp(): return None
            if _proc(child['pid']) != (child['start_ticks'], child['pid']): return None
            if _proc(m['pid'])[0] != m['start_ticks']: return None
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return None
        return (m['boot_id'],m['pid'],m['start_ticks'],m['requested_at'],child['pid'],child['start_ticks'],owner)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _require_guard():
    identity = _guard_identity()
    if identity is None: raise RuntimeError('Live foreground npu-guard with BF16_GROUPED_GUARD_OWNER required')
    return identity


CACHE_NAMES = ('ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR',
 'ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR')
SDK = '/usr/local/Ascend/cann-9.2.0-beta.2'
LD_PATH = str(ROOT/'install/vendors/bf16gmm_transformer/op_api/lib')+':'+':'.join((SDK+'/lib64',SDK+'/lib64/plugin/opskernel',SDK+'/lib64/plugin/nnengine',SDK+'/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64','/usr/local/Ascend/driver/lib64','/usr/local/Ascend/driver/lib64/common','/usr/local/Ascend/driver/lib64/driver'))
PYTHON_PATH = SDK+'/python/site-packages:'+SDK+'/opp/built-in/op_impl/ai_core/tbe:'

def _environment():
    for name, value in ENV.items():
        if os.environ.get(name) != value: raise RuntimeError('Original environment required: '+name)
    for name, value in (('LD_LIBRARY_PATH',LD_PATH),('PYTHONPATH',PYTHON_PATH)):
        if os.environ.get(name) != value: raise RuntimeError('Original loader environment required: '+name)
    for name in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name) not in (None,'','0'): raise RuntimeError('Physical device 0 required')
    for name in ('LD_PRELOAD','PYTHONHOME','FLA_NPU_DISABLE_PTH','TORCH_DEVICE_BACKEND_AUTOLOAD'):
        if os.environ.get(name) not in (None, ''): raise RuntimeError('Unreviewed runtime override: '+name)
    site = Path('/root/dyq/workspace/.venv/lib/python3.12/site-packages')
    if {str(p.resolve()) for p in site.glob('*.pth')} != {p for p in PINS if p.endswith('.pth')}:
        raise RuntimeError('Python startup hooks changed')
    if any(n.rsplit('.',1)[-1] == 'gemm_cpu_bf16' for n in sys.modules):
        raise RuntimeError('Unreviewed GEMM patch module loaded')
    if sys.prefix != '/root/dyq/workspace/.venv': raise RuntimeError('Original Python environment required')
    root = Path(os.environ['BF16_GROUPED_RUN_ROOT']).resolve()
    if any(root == p or p in root.parents for p in (ROOT, ROOT.parent/'bf16_reference_budget2')): raise RuntimeError('Original evidence cannot be a run root')
    for name in CACHE_NAMES:
        p = Path(os.environ[name]).resolve()
        if p != root and root not in p.parents: raise RuntimeError('Cache must be under explicit fresh run root: '+name)
    # Preserve all loader/backend controls across admission and reference execution.
    return (tuple(sorted(os.environ.items())),tuple(sys.path),sys.executable,sys.prefix)


def _new_run(case_id):
    root = Path(os.environ['BF16_GROUPED_RUN_ROOT']).resolve()
    root.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='bf16-grouped-registration1-',dir=root))/case_id
    run.mkdir()
    return run


def _prepare_admission(key):
    identity = _require_guard()
    environment = _environment()
    data = _snapshot(runtime=True)
    private = _load(data)
    _loaded_vendor_pins()
    run_dir = _new_run(key[2])
    envelope = ENVELOPES[key[2]]
    host = private['resource_gate'].capacity()
    disk = shutil.disk_usage(run_dir).free
    if host['effective_available_host_bytes'] < envelope['host_required'] or disk < envelope['disk_required']:
        raise RuntimeError('Original host/disk resource envelope unavailable')
    if _require_guard() != identity: raise RuntimeError('Guard changed before device import')
    import torch, torch_npu
    free,total = torch.npu.mem_get_info()
    admission = dict(_plain(envelope),host=host,disk_free=disk,hbm_free=free,hbm_total=total,
                     stage='before_input_allocation',guard_identity=identity)
    (run_dir/'RESOURCE_ADMISSION.json').write_text(json.dumps(admission,indent=2))
    if free < envelope['device_required']: raise RuntimeError('Original HBM envelope unavailable')
    if _require_guard() != identity or _environment() != environment: raise RuntimeError('Admission identity/environment changed')
    return private,run_dir,identity,environment



CALL_CONTRACTS = MappingProxyType(CALL_CONTRACTS)
CASE_INDICES = MappingProxyType(CASE_INDICES)
ENVELOPES = _freeze(ENVELOPES)

def make(case, baseline='ops-transformer'):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case,baseline): raise UnsupportedCase('Exact one-key BF16 params, indices and tensors required')
    _snapshot()
    key = _key(case,baseline)
    frozen = _frozen_case(key)
    prepared = _prepare_admission(key) if _guard_identity() is not None or _in_guard_group() or os.environ.get('BF16_GROUPED_GUARD_OWNER') else None
    def run(*args,**kwargs):
        nonlocal prepared
        selected,prepared = prepared,None
        if not matches(case,baseline) or _key(case,baseline) != key:
            raise UnsupportedCase('Case changed since construction')
        if selected is None: selected = _prepare_admission(key)
        private,run_dir,identity,environment = selected
        if _require_guard() != identity: raise RuntimeError('Admission belongs to a different guard')
        if _environment() != environment: raise RuntimeError('Environment changed since admission')
        _snapshot(runtime=True)
        _loaded_vendor_pins()
        owned=[]
        def native_factory(proxy):
            child=private['binding'].dense_fp32(proxy)
            owned.append(child)
            return child
        try:
            child = private['adapter'].candidate(frozen,native_factory=native_factory)
            _loaded_vendor_pins(require=True)
            result = child.implementation.run(*args,**kwargs)
            _loaded_vendor_pins(require=True)
            (run_dir/'loaded-maps.txt').write_text(Path('/proc/self/maps').read_text())
            return result
        finally:
            # Also owns the original native candidate if adapter construction fails.
            for native in reversed(owned):
                if native.cleanup is not None: native.cleanup()
    impl = types.SimpleNamespace(run=run,args=None,reset=None,noncomparable_reason=None)
    return Candidate(impl,'ops-transformer::aclnnGroupedMatmulV5',
        'Unchanged approved adapter and original dense_fp32; exact BF16 grouped NT case.',dict(PINS),
        limitations=('Original SDK/vendor, foreground guard and pre-input admission required.',),timing_mode='host_roundtrip')


def wrap(previous):
    def dispatch(case):
        if _key(case) in CASE_PARAMS: return make(case)
        if previous is not None: return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No exact BF16 GroupedGemm route')
    dispatch.previous = previous
    return dispatch
