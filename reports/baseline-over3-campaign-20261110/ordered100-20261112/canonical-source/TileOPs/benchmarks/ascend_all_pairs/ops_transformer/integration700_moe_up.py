"""Exact verified MoE up routes; CPU-lazy, unchanged native cleanup lifetime."""
import ast,dataclasses,hashlib,json,os,types,typing
from pathlib import Path
from types import MappingProxyType
ROOT = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker04moe')
PIN_HASH = '36a03a8506e1199cda7a602742b8bf74a932e5b1bc7bac9f1542462d2f8a9bec'
CASE_PARAMS = {('ops-transformer', 'MoEGroupedGemmFwdOp', 'ds-v3-decode-mlp-up-bfloat16'): '{"a":[[4096,7168],"bfloat16"],"activation":null,"b":[[256,4096,7168],"bfloat16"],"layout":{"contiguous":{"alignment":1,"metadata_kind":"physical_psum","packing":"tight"}},"layout_metadata":[[256],"int32"],"out_dtype":null,"output":[[4096,4096],"bfloat16"]}',
 ('ops-transformer', 'MoEGroupedGemmFwdOp', 'ds-v3-prefill-mlp-up-bfloat16'): '{"a":[[32768,7168],"bfloat16"],"activation":null,"b":[[256,4096,7168],"bfloat16"],"layout":{"contiguous":{"alignment":1,"metadata_kind":"physical_psum","packing":"tight"}},"layout_metadata":[[256],"int32"],"out_dtype":null,"output":[[32768,4096],"bfloat16"]}'}
CASE_INDICES = {('ops-transformer', 'MoEGroupedGemmFwdOp', 'ds-v3-decode-mlp-up-bfloat16'): {'D': 'bfloat16',
                                                                              'E': 256,
                                                                              'K': 7168,
                                                                              'N': 4096,
                                                                              'P': 4096,
                                                                              'R': 4096,
                                                                              'activation': None,
                                                                              'layout': {'contiguous': {'alignment': 1,
                                                                                                        'metadata_kind': 'physical_psum',
                                                                                                        'packing': 'tight'}},
                                                                              'out_dtype': None},
 ('ops-transformer', 'MoEGroupedGemmFwdOp', 'ds-v3-prefill-mlp-up-bfloat16'): {'D': 'bfloat16',
                                                                               'E': 256,
                                                                               'K': 7168,
                                                                               'N': 4096,
                                                                               'P': 32768,
                                                                               'R': 32768,
                                                                               'activation': None,
                                                                               'layout': {'contiguous': {'alignment': 1,
                                                                                                         'metadata_kind': 'physical_psum',
                                                                                                         'packing': 'tight'}},
                                                                               'out_dtype': None}}
def _key(case, baseline='ops-transformer'):
    return baseline, case._call.signature.name, case.id

def _plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping): return {k:_plain(v) for k,v in value.items()}
    if getattr(value,'adt',None)=='MGroupedLayout' and getattr(value,'kind',None)=='contiguous':
        return {value.kind:_plain(value.fields)}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    if type(value) in (str,int,float,bool,type(None)): return value
    raise TypeError('Unsupported parameter representation')

def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)

def matches(case, baseline='ops-transformer'):
    try:
        p=json.loads(CASE_PARAMS[_key(case,baseline)])
        call=case._call
        # Raw Call carries ADTValue in both indices and params. Case.params is
        # its report literal. Check the two interfaces independently.
        for layout in (call.indices['layout'],call.params['layout']):
            if getattr(layout,'adt',None)!='MGroupedLayout' or getattr(layout,'kind',None)!='contiguous': return False
            if _canonical(layout.fields)!=_canonical(p['layout']['contiguous']): return False
        return (_canonical(case.params)==CASE_PARAMS[_key(case,baseline)]
            and _canonical(call.indices)==_canonical(CASE_INDICES[_key(case,baseline)])
            and _canonical(call.tensors)==_canonical({n:p[n] for n in ('a','b','layout_metadata','output')})
            and _canonical(call.params)==_canonical({n:p[n] for n in ('layout','activation','out_dtype')})
            and all(call.present(n) is True for n in ('a','b','layout_metadata','output'))
            and call.present('bias') is False)
    except (AttributeError,KeyError,TypeError,ValueError): return False

def _deepfreeze(value):
    if getattr(value,"adt",None)=="MGroupedLayout": return value
    if isinstance(value,dict): return MappingProxyType({k:_deepfreeze(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)): return tuple(_deepfreeze(v) for v in value)
    return value

def _freeze_case(case):
    from collections import namedtuple
    class FrozenADT(namedtuple('FrozenADTBase','adt kind fields')):
        __slots__=()
        def __getattr__(self,name):
            try:return self.fields[name]
            except KeyError:raise AttributeError(name) from None
    class FrozenCall(namedtuple('FrozenCallBase','signature indices tensors params')):
        __slots__=()
        def present(self,name):return name in self.tensors
    class Snapshot(namedtuple('FrozenCaseBase','id params call')):
        __slots__=()
        @property
        def _call(self):return self.call
    p=json.loads(CASE_PARAMS[_key(case)])
    layout=FrozenADT('MGroupedLayout','contiguous',_deepfreeze(p['layout']['contiguous']))
    raw=dict(layout=layout,activation=p['activation'],out_dtype=p['out_dtype'])
    indices=dict(CASE_INDICES[_key(case)],layout=layout)
    tensors={n:(tuple(p[n][0]),p[n][1]) for n in ('a','b','layout_metadata','output')}
    signature=namedtuple('FrozenSignature','name')(case._call.signature.name)
    call=FrozenCall(signature,_deepfreeze(indices),_deepfreeze(tensors),_deepfreeze(raw))
    return Snapshot(case.id,_deepfreeze(p),call)

def _proc(pid):
    s = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return int(s[19]), int(s[2])

def _guard_identity():
    """Check running foreground group, live supervisor/child and held kernel lock."""
    import fcntl
    try:
        lock = Path('/run/lock/ascend-npu/device-0.lock')
        with lock.open('r') as f:
            m = json.load(f); child = m['child']; owner = os.environ.get('BF16_GEMM_GUARD_OWNER')
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
    if identity is None: raise RuntimeError('Live foreground npu-guard with BF16_GEMM_GUARD_OWNER required')
    return identity

def _pins():
 raw=(ROOT/'REGISTRATION_PINS.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PIN_HASH:raise RuntimeError('Registration pins changed')
 return json.loads(raw)

def _native(case):
 _require_guard()
 vendor='/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer'
 if os.environ.get('ASCEND_CLASSIC_TRANSFORMER_VENDOR')!=vendor or vendor not in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':'):
  raise RuntimeError('Original vendor environment required; use supplied environment.sh')
 if os.environ.get('ASCEND_HOME_PATH')!='/usr/local/Ascend/cann-9.2.0-beta.2':raise RuntimeError('Original SDK required')
 pins=_pins()
 for p,h in pins.items():
  with open(p,'rb') as f: observed=hashlib.file_digest(f,'sha256').hexdigest()
  if observed!=h:raise RuntimeError('Original dependency changed: '+p)
 adapter=ROOT/'adapter.py';raw=adapter.read_bytes()
 module=types.ModuleType('_verified_moe_up_adapter');module.__file__=str(adapter)
 exec(compile(raw,str(adapter),'exec'),module.__dict__)
 return module.make_candidate(case)

def make(case,baseline='ops-transformer'):
 from benchmarks.ascend_baselines import Candidate,UnsupportedCase
 if not matches(case,baseline):raise UnsupportedCase('Exact verified MoE up key and complete manifest contract required')
 frozen=_freeze_case(case)
 # Recover only the original passive dataclass, avoiding device imports in CPU routing.
 path='/root/dyq/workspace/TileOPs/benchmarks/api.py';raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=_pins()[path]:raise RuntimeError('Original API changed')
 cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Implementation')
 ns=dict(dataclasses=dataclasses,Callable=typing.Callable,Optional=typing.Optional)
 exec(compile(ast.Module(body=[cls],type_ignores=[]),path,'exec'),ns)
 child=[_native(frozen) if _guard_identity() is not None else None]
 def run(*args,**kwargs):
  _require_guard()
  if not matches(case,baseline):raise UnsupportedCase('Manifest mutated after factory construction')
  if child[0] is None:child[0]=_native(frozen)
  return child[0].implementation.run(*args,**kwargs)
 def cleanup():
  if child[0] is not None:
   value,child[0]=child[0],None
   if value.cleanup is not None:value.cleanup()
 return Candidate(ns['Implementation'](run),'ops-transformer::aclnnGroupedMatmulV5',
  'Verified original MoE physical-psum bridge and original FP32 native products; cleanup retained until caller cleanup.',
  dict(_pins()),cleanup=cleanup)

def wrap(previous):
 def dispatch(case):
  if _key(case) in CASE_PARAMS:return make(case)
  return previous(case)
 dispatch.previous=previous
 return dispatch
