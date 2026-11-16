"""Verified registration preparation only: exact CPU-lazy GDN local routes."""
import builtins, hashlib, json, os, types, tempfile
from pathlib import Path
from types import MappingProxyType
ROOT = '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package'
MANIFEST = '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_registration10/FROZEN_PINS.json'
MANIFEST_HASH = '835be75dbf6f058188dfe07c16518212a42232d9fa11ed70f48e8e99ac883cf8'
LOCAL_FILES = ('/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/CONTRACTS.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/CHUNK_CASES.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/CASE_RESOURCE_ESTIMATES.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/SOURCE_PINS.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/ADAPTER_INVARIANTS.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/binding.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/direct_acl.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/resource_gate.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/entry.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/RUNTIME_RECIPE.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/RUNTIME_PINS.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/chunk_package/RESOURCE_ESTIMATES.json')
CASE_PARAMS = {('ops-transformer', 'GDNFwdOp', 'q35-27b-16x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,131072,48],"bfloat16"],"cu_seqlens":[[17],"int64"],"cu_seqlens_cpu":[[17],"int64"],"final_state":[[16,48,128,128],"float32"],"g":[[1,131072,48],"bfloat16"],"k":[[1,131072,16,128],"bfloat16"],"o":[[1,131072,48,128],"bfloat16"],"q":[[1,131072,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,131072,48,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-27b-32x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,262144,48],"bfloat16"],"cu_seqlens":[[33],"int64"],"cu_seqlens_cpu":[[33],"int64"],"final_state":[[32,48,128,128],"float32"],"g":[[1,262144,48],"bfloat16"],"k":[[1,262144,16,128],"bfloat16"],"o":[[1,262144,48,128],"bfloat16"],"q":[[1,262144,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,262144,48,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-2b-32x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,262144,16],"bfloat16"],"cu_seqlens":[[33],"int64"],"cu_seqlens_cpu":[[33],"int64"],"final_state":[[32,16,128,128],"float32"],"g":[[1,262144,16],"bfloat16"],"k":[[1,262144,16,128],"bfloat16"],"o":[[1,262144,16,128],"bfloat16"],"q":[[1,262144,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,262144,16,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-35b-16x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,131072,32],"bfloat16"],"cu_seqlens":[[17],"int64"],"cu_seqlens_cpu":[[17],"int64"],"final_state":[[16,32,128,128],"float32"],"g":[[1,131072,32],"bfloat16"],"k":[[1,131072,16,128],"bfloat16"],"o":[[1,131072,32,128],"bfloat16"],"q":[[1,131072,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,131072,32,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-35b-32x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,262144,32],"bfloat16"],"cu_seqlens":[[33],"int64"],"cu_seqlens_cpu":[[33],"int64"],"final_state":[[32,32,128,128],"float32"],"g":[[1,262144,32],"bfloat16"],"k":[[1,262144,16,128],"bfloat16"],"o":[[1,262144,32,128],"bfloat16"],"q":[[1,262144,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,262144,32,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-397b-tp1-16x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,131072,64],"bfloat16"],"cu_seqlens":[[17],"int64"],"cu_seqlens_cpu":[[17],"int64"],"final_state":[[16,64,128,128],"float32"],"g":[[1,131072,64],"bfloat16"],"k":[[1,131072,16,128],"bfloat16"],"o":[[1,131072,64,128],"bfloat16"],"q":[[1,131072,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,131072,64,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-397b-tp1-32x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,262144,64],"bfloat16"],"cu_seqlens":[[33],"int64"],"cu_seqlens_cpu":[[33],"int64"],"final_state":[[32,64,128,128],"float32"],"g":[[1,262144,64],"bfloat16"],"k":[[1,262144,16,128],"bfloat16"],"o":[[1,262144,64,128],"bfloat16"],"q":[[1,262144,16,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,262144,64,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-397b-tp2-32x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,262144,32],"bfloat16"],"cu_seqlens":[[33],"int64"],"cu_seqlens_cpu":[[33],"int64"],"final_state":[[32,32,128,128],"float32"],"g":[[1,262144,32],"bfloat16"],"k":[[1,262144,8,128],"bfloat16"],"o":[[1,262144,32,128],"bfloat16"],"q":[[1,262144,8,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,262144,32,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-symmetric-16x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,131072,32],"bfloat16"],"cu_seqlens":[[17],"int64"],"cu_seqlens_cpu":[[17],"int64"],"final_state":[[16,32,128,128],"float32"],"g":[[1,131072,32],"bfloat16"],"k":[[1,131072,32,128],"bfloat16"],"o":[[1,131072,32,128],"bfloat16"],"q":[[1,131072,32,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,131072,32,128],"bfloat16"]}',
 ('ops-transformer', 'GDNFwdOp', 'q35-symmetric-32x8192-bfloat16'): '{"allow_neg_eigval":false,"beta":[[1,262144,32],"bfloat16"],"cu_seqlens":[[33],"int64"],"cu_seqlens_cpu":[[33],"int64"],"final_state":[[32,32,128,128],"float32"],"g":[[1,262144,32],"bfloat16"],"k":[[1,262144,32,128],"bfloat16"],"o":[[1,262144,32,128],"bfloat16"],"q":[[1,262144,32,128],"bfloat16"],"scale":null,"state_v_first":false,"use_beta_sigmoid_in_kernel":false,"use_gate_in_kernel":false,"use_qk_l2norm_in_kernel":false,"v":[[1,262144,32,128],"bfloat16"]}'}
ENV = {'ASCEND_CUSTOM_OPP_PATH': '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer',
 'ASCEND_HOME_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'ASCEND_OPP_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2/opp',
 'ASCEND_TOOLKIT_HOME': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'FLA_NPU_DISABLE_PTH': '1',
 'GDN_CASE_OWNER': 'baseline700-20261026/gdn_local_native10',
 'PYTHONDONTWRITEBYTECODE': '1',
 'PYTHONNOUSERSITE': '1',
 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}
LD_PATH = '/usr/local/Ascend/cann-9.2.0-beta.2/lib64:/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_api/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib:/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib'
PYTHON_PATH = '/root/dyq/workspace/.venv/lib/python3.12/site-packages:/usr/local/Ascend/cann-9.2.0-beta.2/python/site-packages:/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe'
LOADED_VENDOR = {'/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_api/lib/libcust_opapi.so': '283c1f3c6e567f956487078b3714c2005f463a238d84ffc25a5120988158c340',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so': '86ba906ef7caa05b817a197133b57afbf190686a3752037942fba426e3810ed8',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_proto/es/lib/linux/x86_64/libes_transformer_cust.so': '3103eb0c3b96d594b2bc8f48d630f2939f12fc00a037f4d5f322a58d8938e5fb',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so': '06b95ad9e297ea74a74e51824aa63359a1e7b7a2fcdffa7778f33502a5ea5576',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascend_dump.so': 'c23cc357303d1fdd5540db7a9c519344fa3b922372b0bf370e60f7e48a778be2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascendcl.so': '4b270e626aad5d6a1e2b5444c60203185cf4684e06843573afb78e539b385fb2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libnnopbase.so': 'f362a84c4c144f0a2cb1d8dc5266edf1dcb009f6843f0025139675c31afbd962',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopapi_math.so': '611d12ac698d797a5f81f2416072bb0428a898bbe1a53a5ff762feb5cfb49f29',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libprofapi.so': 'b3d3b9a2992df7214cffb31a34eb7772957af6f4a590d7c86debda5b2b7adc0a',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libruntime.so': '1f64c1271a3eaae2a5429289112b229e20b7b8a0641cf8df5ab6c12bd24939c2'}
REQUIRED_LOADED = ('/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_api/lib/libcust_opapi.so',
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
        raise RuntimeError('Frozen GDN dependency changed: ' + str(path))
    return raw

# The complete manifest is fixed by the published module, never trusted dynamically.
PINS = MappingProxyType(json.loads(_checked(MANIFEST, MANIFEST_HASH)))
CASE_PARAMS = MappingProxyType(CASE_PARAMS)
ENV = MappingProxyType(ENV)
LOADED_VENDOR = MappingProxyType(LOADED_VENDOR)

def _snapshot(runtime=False):
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

def matches(case, baseline='ops-transformer'):
    try:
        return json.dumps(case.params, sort_keys=True, separators=(',', ':'), allow_nan=False) == CASE_PARAMS.get(_key(case, baseline))
    except (TypeError, ValueError):
        return False

def _load(data):
    # No module is cached or inserted into sys.modules. Resolve the two private
    # imports internally, even if an unrelated caller poisoned those names.
    private = {}; original = builtins.__import__
    class FrozenPath(type(Path())):
        def read_bytes(self):
            if str(self) in data: return data[str(self)]
            return super().read_bytes()
        def read_text(self, *args, **kwargs):
            if str(self) in data: return data[str(self)].decode('utf-8')
            return super().read_text(*args, **kwargs)
    pathlib_proxy = types.ModuleType('pathlib'); pathlib_proxy.Path = FrozenPath
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in ('direct_acl', 'resource_gate'): return private[name]
        if level == 0 and name == 'pathlib': return pathlib_proxy
        return original(name, globals, locals, fromlist, level)
    for name in ('direct_acl', 'resource_gate', 'binding'):
        path = str(ROOT/(name+'.py')); mod = types.ModuleType('_gdn_local_verified_'+name)
        mod.__file__ = path
        mod.__dict__['__builtins__'] = dict(vars(builtins), __import__=private_import)
        exec(compile(data[path], path, 'exec'), mod.__dict__)
        private[name] = mod
    return private

def _environment():
    for name in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name) not in (None, '', '0'):
            raise RuntimeError('Original physical device 0 required: '+name)
    for name, value in ENV.items():
        if os.environ.get(name) != value: raise RuntimeError('Original environment required: '+name)
    if os.environ.get('LD_LIBRARY_PATH') != LD_PATH: raise RuntimeError('Original LD_LIBRARY_PATH required')
    if os.environ.get('PYTHONPATH') != PYTHON_PATH: raise RuntimeError('Original PYTHONPATH required')

def _new_run(case_id):
    root = Path(os.environ['GDN_RUN_DIR']).resolve()
    if root == ROOT.parent or ROOT.parent in root.parents:
        raise RuntimeError('Cannot overwrite original native evidence')
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='gdn-local-registration10-', dir=root))/case_id
    run.mkdir()
    return run

def _hbm_admission(torch, run, admission):
    torch.npu.set_device(0)
    free, total = torch.npu.mem_get_info()
    estimate = admission['required_envelope']['estimated_device_budget_bytes']
    safety = 2*1024**3
    value = dict(free_hbm_bytes=free,total_hbm_bytes=total,estimated_device_budget_bytes=estimate,
                 safety_margin_bytes=safety,admitted=free >= estimate+safety)
    (run/'HBM_ADMISSION.json').write_text(json.dumps(value,indent=2))
    if not value['admitted']: raise RuntimeError('Insufficient actual free HBM with 2 GiB safety margin')

def _guard_identity():
    """Return identity only for this foreground guarded process, without device imports."""
    try:
        metadata = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    except (OSError, ValueError):
        return None
    child = metadata.get('child', {})
    owner = os.environ.get('GDN_CASE_OWNER')
    if (not owner or metadata.get('owner') != owner or metadata.get('phase') != 'running'
            or child.get('pid') != os.getpgrp()):
        return None
    return (metadata.get('boot_id'), metadata.get('pid'), metadata.get('start_ticks'),
            metadata.get('requested_at'), child.get('pid'), child.get('start_ticks'), owner)


def _prepare_admission(key):
    private = _load(_snapshot())
    private['direct_acl'].require_guard()
    identity = _guard_identity()
    if identity is None:
        raise RuntimeError('Current foreground guard identity required')
    _environment()
    data = _snapshot(runtime=True)
    private = _load(data)
    private['direct_acl'].require_guard()
    _loaded_vendor_pins()
    run_dir = _new_run(key[2])
    gate = private['resource_gate']
    admission = gate.require_capacity(gate.Path(str(ROOT)), run_dir)
    import torch, torch_npu
    _hbm_admission(torch, run_dir, admission)
    if _guard_identity() != identity:
        raise RuntimeError('Guard identity changed during resource admission')
    (run_dir/'GUARD_ADMISSION_IDENTITY.json').write_text(json.dumps(identity))
    return private, run_dir, identity


def make(case, baseline='ops-transformer'):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case, baseline): raise UnsupportedCase('Exact GDN key and all canonical parameters required')
    _snapshot()
    key = _key(case, baseline)
    frozen = types.SimpleNamespace(id=key[2], params=json.loads(CASE_PARAMS[key]))
    # The original checker constructs Candidate before materializing case.inputs.
    # Preserve that admission phase; charging the full envelope again after
    # inputs/reference allocation would double-count already allocated storage.
    prepared = _prepare_admission(key) if _guard_identity() is not None else None
    def run(*args, **kwargs):
        nonlocal prepared
        if not matches(case, baseline): raise UnsupportedCase('Case parameters changed after construction')
        if prepared is None:
            # A CPU-only caller may construct outside a reservation. Such a
            # candidate must obtain a fresh full admission in its guarded run.
            selected = _prepare_admission(key)
        else:
            selected, prepared = prepared, None  # one admission, one invocation
        private, run_dir, identity = selected
        private['direct_acl'].require_guard()
        if _guard_identity() != identity:
            raise RuntimeError('Resource admission belongs to a different guard')
        _environment()
        _snapshot(runtime=True)  # dependency changes while reference runs fail closed
        _loaded_vendor_pins()
        child = private['binding'].build(frozen)
        _loaded_vendor_pins(require=True)
        try:
            result = child.implementation.run(*args, **kwargs)
            _loaded_vendor_pins(require=True)
            (run_dir/'loaded-maps.txt').write_text(Path('/proc/self/maps').read_text())
            return result
        finally:
            if child.cleanup is not None: child.cleanup()
    impl = types.SimpleNamespace(run=run,args=None,reset=None,noncomparable_reason=None)
    return Candidate(impl,'ops-transformer::aclnnChunkGatedDeltaRule',
        'Unchanged verified original whole chunk API, exact ten local cases and both outputs.',dict(PINS),
        limitations=('Pinned local SDK, original environment, foreground guard and fresh host/disk/HBM admission required.',),
        timing_mode='host_roundtrip')

def wrap(previous):
    def dispatch(case):
        if _key(case) in CASE_PARAMS: return make(case)
        if previous is not None: return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No exact GDN local route')
    dispatch.previous = previous
    return dispatch
