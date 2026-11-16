from pathlib import Path
from types import SimpleNamespace
import ctypes as C,os,json,hashlib
from benchmarks.ascend_baselines import Candidate,WORKSPACE,_owned_native
from benchmarks.api import Implementation
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-aot-original-v2')
def source_hashes(*paths):
 out={}
 for p in paths:
  p=Path(p)
  for f in ([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()]):out[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return out
class _MathCall:

    def __init__(self, api, arguments, *, library='ops-math', host_library=None):
        import torch
        self.torch = torch
        self.pending = []
        suffix = {'ops-math': 'math', 'ops-nn': 'nn', 'ops-transformer': 'transformer'}[library]
        self.vendor = Path(os.environ.get(f'ASCEND_CLASSIC_{suffix.upper()}_VENDOR', str(WORKSPACE / '.cache/ascend-classic/native-pilot-20261020' / library / f'install/vendors/classicpilot_{suffix}'))).resolve()
        configured = {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if p}
        if self.vendor not in configured:
            raise RuntimeError(f'Select the isolated {library} vendor before importing Torch.')
        self.binary = Path(host_library).resolve() if host_library else self.vendor / 'op_api/lib/libcust_opapi.so'
        self.lib = _owned_native(self.binary, [api, api + 'GetWorkspaceSize'])
        self.api = api
        self.size = getattr(self.lib, api + 'GetWorkspaceSize')
        self.size.argtypes = arguments + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        self.size.restype = C.c_int
        self.launch = getattr(self.lib, api)
        self.launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        self.launch.restype = C.c_int
        self.base = C.CDLL('libnnopbase.so')
        self.base.aclCreateTensor.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64), C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
        self.base.aclCreateTensor.restype = C.c_void_p
        self.base.aclDestroyTensor.argtypes = [C.c_void_p]
        self.base.aclDestroyTensor.restype = C.c_int
        self.base.aclCreateIntArray.argtypes = [C.POINTER(C.c_int64), C.c_uint64]
        self.base.aclCreateIntArray.restype = C.c_void_p
        self.base.aclDestroyIntArray.argtypes = [C.c_void_p]
        self.base.aclDestroyIntArray.restype = C.c_int
        self.codes = {torch.float32: 0, torch.float16: 1, torch.int32: 3, torch.int64: 9, torch.complex64: 16, torch.bfloat16: 27, torch.float8_e4m3fn: 36}

    @staticmethod
    def check(rc, api):
        if rc:
            raise RuntimeError(f'{api} returned {rc}')

    def record(self):
        item = {'tensors': [], 'arrays': [], 'keep': []}
        self.pending.append(item)
        return item

    def tensor(self, tensor, record, *, format_code=2):
        if not tensor.is_contiguous():
            raise ValueError('This host bridge expects packed tensors.')
        shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (C.c_int64 * tensor.ndim)(*tensor.stride())
        record['keep'].extend((tensor, shape, stride))
        ptr = self.base.aclCreateTensor(shape, tensor.ndim, self.codes[tensor.dtype], stride, 0, format_code, shape, tensor.ndim, tensor.data_ptr())
        if not ptr:
            raise RuntimeError('aclCreateTensor returned null')
        record['tensors'].append(ptr)
        return ptr

    def array(self, values, record):
        storage = (C.c_int64 * len(values))(*values)
        record['keep'].append(storage)
        ptr = self.base.aclCreateIntArray(storage, len(values))
        if not ptr:
            raise RuntimeError('aclCreateIntArray returned null')
        record['arrays'].append(ptr)
        return ptr

    def call(self, arguments, record, device):
        size, executor = (C.c_uint64(), C.c_void_p())
        self.check(self.size(*arguments, C.byref(size), C.byref(executor)), self.api + 'GetWorkspaceSize')
        workspace = self.torch.empty(size.value, device=device, dtype=self.torch.uint8)
        record['keep'].extend((workspace, executor))
        self.check(self.launch(workspace.data_ptr() if size.value else None, size.value, executor, self.torch.npu.current_stream().npu_stream), self.api)

    def cleanup(self):
        self.torch.npu.synchronize()
        while self.pending:
            record = self.pending.pop()
            for p in record['tensors']:
                self.check(self.base.aclDestroyTensor(p), 'aclDestroyTensor')
            for p in record['arrays']:
                self.check(self.base.aclDestroyIntArray(p), 'aclDestroyIntArray')

    def hashes(self, root):
        sources = [p for p in root.rglob('*') if p.suffix in ('.cpp', '.h', '.cc', '.py', '.json')]
        deployed = [p for p in self.vendor.rglob('*') if p.is_file() and p.suffix in ('.so', '.o', '.json')]
        return source_hashes(Path(__file__), self.binary, *sources, *deployed)
def dense_fp32(case):
    import torch
    from ops_transformer_gmmv5_aot_support import _MathCall
    from benchmarks.ascend_baselines import Candidate
    from ops_transformer_gmmv5_aot_support import source_hashes
    from benchmarks.api import Implementation
    vendor = N / 'private-v1/vendors/bf16gmm_transformer'
    b = _MathCall('aclnnGroupedMatmulV5', [C.c_void_p] * 12 + [C.c_int64] * 4 + [C.c_void_p] * 4, library='ops-transformer', host_library=vendor / 'op_api/lib/libcust_opapi.so')
    base = b.base
    base.aclCreateTensorList.argtypes = [C.POINTER(C.c_void_p), C.c_uint64]
    base.aclCreateTensorList.restype = C.c_void_p
    base.aclDestroyTensorList.argtypes = [C.c_void_p]
    base.aclDestroyTensorList.restype = C.c_int
    lists = []

    def tensorlist(ts, record):
        ptrs = [b.tensor(t, record) for t in ts]
        arr = (C.c_void_p * len(ptrs))(*ptrs)
        record['keep'].append(arr)
        out = base.aclCreateTensorList(arr, len(ptrs))
        if not out:
            raise RuntimeError('aclCreateTensorList failed')
        for ptr in ptrs:
            record['tensors'].remove(ptr)
        lists.append(out)
        return out

    def run(a, w, bias=None):
        original_dtype = a.dtype
        a = a.float()
        w = w.float()
        if case._call.signature.name == 'BmmFwdOp':
            lhs = [t.contiguous() for t in a]
            rhs = [t.contiguous() for t in w]
        else:
            lhs = [(a.T if case.params['trans_a'] else a).contiguous()]
            rhs = [(w.T if case.params['trans_b'] else w).contiguous()]
        ys = [torch.empty((x.shape[0], y.shape[1]), dtype=a.dtype, device=a.device) for x, y in zip(lhs, rhs, strict=True)]
        record = b.record()
        args = [tensorlist(lhs, record), tensorlist(rhs, record), None if bias is None else tensorlist([bias.float().contiguous()], record)] + [None] * 9 + [0, -1, 0, 0, None, tensorlist(ys, record), None, None]
        b.call(args, record, a.device)
        return (torch.stack(ys) if case._call.signature.name == 'BmmFwdOp' else ys[0]).to(original_dtype)

    def cleanup():
        torch.npu.synchronize()
        while lists:
            b.check(base.aclDestroyTensorList(lists.pop()), 'aclDestroyTensorList')
        b.cleanup()
    files = [Path(__file__), vendor / 'op_api/lib/libcust_opapi.so'] + [f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py', '.cpp', '.h') and ('backup' not in str(f))]
    return Candidate(Implementation(run), 'ops-transformer::aclnnGroupedMatmulV5', 'One original no-group TensorList call for every batch; canonical dtypes, pure transpose/packing/output stack. No extra arithmetic.', source_hashes(*files), cleanup=cleanup)
def make_runner(operator, params, *, native_factory, tensor_api=None):
    """Return (run, native_candidate); factory is explicit and never inferred.

    params must be the exact manifest parameters, including T and bias presence.
    NN supported for contract tests; assigned Grouped keys are NT only. TN/TT stop.
    """
    p = dict(params)
    if p.get('T') != 'float16':
        raise ValueError('This exact scope is explicitly FP16 only')
    if operator not in ('BmmFwdOp', 'Conv1dFwdOp', 'GroupedGemmFwdOp'):
        raise NotImplementedError(operator)
    if operator == 'GroupedGemmFwdOp' and p['transpose_a']:
        raise NotImplementedError('TN/TT outside reviewed thin-binding scope')
    if operator == 'Conv1dFwdOp' and (p['groups'] != 1 or not all((isinstance(p[k], int) for k in ('stride', 'padding', 'dilation')))):
        raise NotImplementedError('Only reviewed integer ungrouped Conv1d packing')
    if not callable(native_factory):
        raise TypeError('Explicit reviewed native factory is required')
    if tensor_api is None:
        import torch as tensor_api
    t = tensor_api
    native_op = 'BmmFwdOp' if operator == 'BmmFwdOp' else 'GemmFwdOp'
    proxy = SimpleNamespace(_call=SimpleNamespace(signature=SimpleNamespace(name=native_op)), params={'trans_a': False, 'trans_b': False})
    inner = native_factory(proxy)
    product = inner.implementation.run

    def check(x, shape, dtype=getattr(t, p['T'])):
        if tuple(x.shape) != tuple(shape) or x.dtype != dtype:
            raise ValueError(f'Tensor contract mismatch: {x.shape}, {x.dtype}')

    def result(y, shape):
        check(y, shape)
        return y

    def bmm(a, b):
        check(a, (p['B'], p['M'], p['K']))
        check(b, (p['B'], p['K'], p['N']))
        return result(product(a, b), (p['B'], p['M'], p['N']))

    def conv(x, w, bias=None):
        check(x, (p['N'], p['C_in'], p['L_in']))
        check(w, (p['C_out'], p['C_in'], p['kW']))
        if (bias is not None) != p['bias']:
            raise ValueError('Manifest bias presence mismatch')
        if bias is not None:
            check(bias, (p['C_out'],))
        st, pad, dil = (p['stride'], p['padding'], p['dilation'])
        x = t.nn.functional.pad(x, (pad, pad))
        length = (x.shape[2] - dil * (w.shape[2] - 1) - 1) // st + 1
        windows = x.as_strided((x.shape[0], x.shape[1], length, w.shape[2]), x.stride()[:2] + (x.stride()[2] * st, x.stride()[2] * dil))
        lhs = windows.permute((0, 2, 1, 3)).contiguous().reshape(-1, w[0].numel())
        rhs = w.reshape(w.shape[0], -1).T.contiguous()
        y = product(lhs, rhs, bias)
        result(y, (p['N'] * length, p['C_out']))
        return y.reshape((x.shape[0], length, w.shape[0])).movedim(-1, 1).contiguous()

    def grouped(a, b, sizes, offsets):
        check(a, (p['M'], p['K']))
        check(b, (p['G'], p['N'], p['K']) if p['transpose_b'] else (p['G'], p['K'], p['N']))
        check(sizes, (p['G'],), t.int32)
        check(offsets, (p['G'],), t.int32)
        ns, starts = (sizes.tolist(), offsets.tolist())
        end = 0
        for n, start in zip(ns, starts, strict=True):
            if type(n) is not int or type(start) is not int or n < 0 or (start != end):
                raise ValueError('sizes/offsets violate canonical prefix invariant')
            end += n
        if end != p['M']:
            raise ValueError('sizes do not cover complete output')
        out = t.empty((p['M'], p['N']), dtype=a.dtype, device=a.device)
        for i, (n, start) in enumerate(zip(ns, starts, strict=True)):
            if n == 0:
                continue
            rhs = b[i].T if p['transpose_b'] else b[i]
            y = product(a[start:start + n].contiguous(), rhs.contiguous())
            out[start:start + n] = result(y, (n, p['N']))
        return out
    return ({'BmmFwdOp': bmm, 'Conv1dFwdOp': conv, 'GroupedGemmFwdOp': grouped}[operator], inner)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='GroupedGemmFwdOp' and case.id=='moe-fwd-float16'
  p=dict(case._call.indices);assert p['T']=='float16' and not p['transpose_a'] and p['transpose_b']
  manifest=N/'TRANSFORMER_GMMV5_PACKAGE.json';m=json.loads(manifest.read_text());sources={}
  for name,v in m['files'].items():
   f=Path(m['vendor'])/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256'];sources[str(f)]=v['sha256']
  for f in [Path(__file__),manifest,Path(__file__).with_name('TRANSFORMER_GMMV5_API_FORMAL_PREREQUISITES.json')]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  def run(*args,**kwargs):
   owned=[]
   def native_factory(proxy):
    child=dense_fp32(proxy);owned.append(child);return child
   try:
    grouped,inner=make_runner(case._call.signature.name,p,native_factory=native_factory)
    return grouped(*args,**kwargs)
   finally:
    for native in reversed(owned):
     if native.cleanup is not None:native.cleanup()
  return Candidate(Implementation(run),'ops-transformer::aclnnGroupedMatmulV5','Original exact FP16 NT grouped adapter and complete original dense_fp32 TensorList/widen/cast/cleanup inside each invocation; source/runtime admission checked outside measured value work.',sources,timing_mode='host_roundtrip')
 return build
