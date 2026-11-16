"""CPU-only source/signature/shape checks; never a numerical or device receipt.

Run with python -B -m benchmarks.ascend_all_pairs.tla.static_check from TileOPs.
The fake torch/tilelang modules below only carry metadata. No vendor kernel body,
compiler, tensor computation, accelerator import or external runtime is executed.
"""
from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import types

from . import PROVIDERS, native
from .plans import assess
from .metadata import parameters

HERE=Path(__file__).resolve().parent
CALLS=[]
LOADS=[]


class Expr:
    def __init__(self,s): self.s=str(s)
    def _bin(self,o,op): return Expr(f'({self.s}{op}{o.s if isinstance(o,Expr) else o})')
    __mul__=lambda self,o:self._bin(o,'*')
    __rmul__=__mul__
    __add__=lambda self,o:self._bin(o,'+')
    __radd__=__add__
    __sub__=lambda self,o:self._bin(o,'-')
    __floordiv__=lambda self,o:self._bin(o,'//')
    __truediv__=lambda self,o:self._bin(o,'/')
    __mod__=lambda self,o:self._bin(o,'%')
    def __repr__(self): return self.s


class DType(str):
    def __repr__(self): return 'torch.'+str(self)


def normdtype(dtype):
    return DType({'float':'float32','half':'float16','int':'int32','long':'int64'}.get(str(dtype),str(dtype)))


def _shape(args):
    return tuple(args[0]) if len(args)==1 and isinstance(args[0],(tuple,list)) else tuple(args)


def broadcast(a,b):
    n=max(len(a),len(b)); aa=(1,)*(n-len(a))+a; bb=(1,)*(n-len(b))+b
    assert all(x==y or x==1 or y==1 for x,y in zip(aa,bb)),(a,b)
    return tuple(max(x,y) for x,y in zip(aa,bb))


class Tensor:
    def __init__(self,shape=(),dtype='float32',origin=(),values=None):
        self.shape=tuple(shape); self.dtype=normdtype(dtype); self.device='metadata-only'
        self.origin=frozenset(origin); self.values=values
    @property
    def ndim(self):return len(self.shape)
    @property
    def T(self):return self.permute(*range(self.ndim-1,-1,-1))
    def dim(self):return self.ndim
    def size(self,i=None):return self.shape if i is None else self.shape[i]
    def numel(self):return math.prod(self.shape)
    def stride(self,i=None):
        out=tuple(math.prod(self.shape[j+1:]) for j in range(self.ndim))
        return out if i is None else out[i]
    def is_contiguous(self):return True
    def contiguous(self):return self
    def clone(self):return Tensor(self.shape,self.dtype,values=self.values)
    def detach(self):return self
    def to(self,*args,**kw):
        dtype=kw.get('dtype',next((a for a in args if isinstance(a,DType)),self.dtype))
        return self if dtype==self.dtype else Tensor(self.shape,dtype,values=self.values)
    def float(self):return self.to(dtype=DType('float32'))
    def half(self):return self.to(dtype=DType('float16'))
    def int(self):return self.to(dtype=DType('int32'))
    def long(self):return self.to(dtype=DType('int64'))
    def bool(self):return self.to(dtype=DType('bool'))
    def reshape(self,*args):
        shape=list(_shape(args)); assert shape.count(-1)<=1
        if -1 in shape:shape[shape.index(-1)]=self.numel()//-math.prod(shape)
        assert math.prod(shape)==self.numel(),(self.shape,shape)
        return Tensor(shape,self.dtype,self.origin,self.values)
    def view(self,*args):
        if len(args)==1 and isinstance(args[0],DType):
            sizes={'int64':8,'int32':4,'int16':2,'bool':1,'uint8':1,'int8':1}
            shape=list(self.shape);shape[-1]=shape[-1]*sizes[self.dtype]//sizes[args[0]]
            return Tensor(shape,args[0],self.origin)
        return self.reshape(*args)
    def flatten(self,start_dim=0,end_dim=-1):
        end_dim%=self.ndim
        return self.reshape(self.shape[:start_dim]+(math.prod(self.shape[start_dim:end_dim+1]),)+self.shape[end_dim+1:])
    def permute(self,*args):
        dims=_shape(args); assert sorted(dims)==list(range(self.ndim)),(dims,self.shape)
        return Tensor(tuple(self.shape[i] for i in dims),self.dtype,self.origin)
    def transpose(self,a,b):
        dims=list(range(self.ndim));dims[a],dims[b]=dims[b],dims[a]
        return self.permute(dims)
    def unsqueeze(self,dim):
        if dim<0:dim+=self.ndim+1
        return Tensor(self.shape[:dim]+(1,)+self.shape[dim:],self.dtype,self.origin)
    def squeeze(self,dim=None):
        shape=list(self.shape)
        if dim is None:shape=[n for n in shape if n!=1]
        elif shape[dim]==1:shape.pop(dim)
        return Tensor(shape,self.dtype,self.origin)
    def expand(self,*args):
        shape=_shape(args); base=(1,)*(len(shape)-self.ndim)+self.shape
        shape=tuple(a if b==-1 else b for a,b in zip(base,shape))
        assert broadcast(base,shape)==shape,(base,shape)
        return Tensor(shape,self.dtype,self.origin)
    def flip(self,dims):return Tensor(self.shape,self.dtype)
    def unfold(self,dim,size,step):
        shape=list(self.shape);shape[dim]=(shape[dim]-size)//step+1;shape.append(size)
        return Tensor(shape,self.dtype,self.origin)
    def repeat_interleave(self,repeats,dim):
        shape=list(self.shape);shape[dim]*=repeats
        return Tensor(shape,self.dtype)
    def __getitem__(self,key):
        if not isinstance(key,tuple):key=(key,)
        parts=list(key)
        if Ellipsis in parts:
            i=parts.index(Ellipsis);parts[i:i+1]=[slice(None)]*(self.ndim-sum(k is not None and k is not Ellipsis for k in parts))
        parts += [slice(None)]*(self.ndim-sum(k is not None for k in parts))
        shape=[];j=0
        for k in parts:
            if k is None:shape.append(1);continue
            n=self.shape[j];j+=1
            if isinstance(k,slice):shape.append(len(range(*k.indices(n))))
            elif isinstance(k,int):assert -n<=k<n,(k,self.shape)
            elif isinstance(k,Tensor):shape.extend(k.shape)
            else:raise TypeError(f'Unsupported metadata index {k}')
        val=None
        if self.values is not None and len(key)==1 and isinstance(key[0],(int,slice)):
            val=self.values[key[0]]
        return Tensor(shape,self.dtype,self.origin,val)
    def __setitem__(self,key,value):
        assert not self.origin, f'Adapter writes input {self.origin}'
        target=self[key]
        if isinstance(value,Tensor):assert broadcast(value.shape,target.shape)==target.shape
    def _op(self,other,dtype=None):
        shape=broadcast(self.shape,other.shape) if isinstance(other,Tensor) else self.shape
        values=None
        return Tensor(shape,dtype or self.dtype,values=values)
    __add__=lambda self,o:self._op(o)
    __radd__=__add__
    __sub__=__add__
    __rsub__=__add__
    def __mul__(self,o):
        result=self._op(o)
        if self.values is not None and isinstance(o,(int,float)):
            result.values=[v*o for v in self.values] if isinstance(self.values,list) else self.values*o
        return result
    __rmul__=__mul__
    __truediv__=__add__
    __rtruediv__=__add__
    __pow__=__add__
    __rpow__=__add__
    __and__=__add__
    __or__=__add__
    __lshift__=__add__
    __rshift__=__add__
    __lt__=lambda self,o:self._op(o,'bool')
    __le__=__lt__
    __ge__=__lt__
    __gt__=__lt__
    def bitwise_or_(self,o):assert not self.origin;return self
    def copy_(self,o):assert not self.origin;assert self.shape==o.shape;return self
    def zero_(self):assert not self.origin;return self
    def clamp(self,*args):return Tensor(self.shape,self.dtype)
    def sin(self):return Tensor(self.shape,self.dtype)
    cos=sin
    def tolist(self):
        if self.values is None:raise ValueError(f'No metadata values for {self.shape}')
        return self.values
    def item(self):
        if self.values is None:raise ValueError('No scalar metadata value')
        return self.values
    def new_empty(self,shape):return Tensor(shape,self.dtype)


class Spec:
    def __init__(self,shape,dtype):self.shape=tuple(shape);self.dtype=normdtype(dtype)


def ast_kernel_writes(fn):
    return _kernel_writes(fn.__code__.co_filename, fn.__code__.co_firstlineno, fn.__name__)


@functools.lru_cache(None)
def _kernel_writes(filename, firstlineno, name):
    # A native input is writable only if its formal buffer is a destination in
    # the original AST. Output buffers are separately allocated by fake_compile.
    tree=source_ast(filename)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
              and n.name==name and n.lineno in (firstlineno,firstlineno+1))
    writes=set()
    def root(n):
        while isinstance(n,(ast.Subscript,ast.Attribute)):n=n.value
        return n.id if isinstance(n,ast.Name) else None
    for n in ast.walk(node):
        if not isinstance(n,ast.Call):continue
        name=ast.unparse(n.func)
        if name=='T.copy' and len(n.args)>=2:writes.add(root(n.args[1]))
        if name in ('T.tile.atomic_add','T.atomic_add') and n.args:writes.add(root(n.args[0]))
    return writes


class Kernel:
    def __init__(self,fn,opts):
        self.fn=fn; self.params=list(inspect.signature(fn).parameters.items())
        n=len(self.params)
        self.outputs=opts.get('out_idx') or []
        if isinstance(self.outputs,int):self.outputs=[self.outputs]
        self.outputs=[i%n for i in self.outputs]
        ws=opts.get('workspace_idx') or []
        self.workspaces=[i%n for i in ([ws] if isinstance(ws,int) else ws)]
        self.writes=ast_kernel_writes(fn)
    def __call__(self,*args):
        ins=[(i,name,p.annotation) for i,(name,p) in enumerate(self.params)
             if i not in self.outputs+self.workspaces]
        assert len(args)==len(ins),(self.fn.__code__.co_filename,self.fn.__name__,len(args),len(ins))
        symbols={}
        for (_,name,spec),arg in zip(ins,args):
            assert isinstance(arg,Tensor),(name,type(arg))
            assert len(spec.shape)==arg.ndim,(name,spec.shape,arg.shape)
            assert spec.dtype==arg.dtype,(name,spec.dtype,arg.dtype)
            for n,got in zip(spec.shape,arg.shape):
                if isinstance(n,Expr) and n.s.isidentifier():
                    if n.s in symbols:assert symbols[n.s]==got
                    symbols[n.s]=got
        def value(n):return int(eval(n.s,{'__builtins__':{}},symbols)) if isinstance(n,Expr) else int(n)
        for (_,name,spec),arg in zip(ins,args):
            assert tuple(map(value,spec.shape))==arg.shape,(name,spec.shape,arg.shape)
            if name in self.writes:assert not arg.origin, f'Native kernel writes protected input {name}: {arg.origin}'
        CALLS.append((self.fn.__code__.co_filename,self.fn.__name__))
        outputs=[Tensor(tuple(map(value,self.params[i][1].annotation.shape)),self.params[i][1].annotation.dtype)
                 for i in self.outputs]
        return outputs[0] if len(outputs)==1 else outputs


def fake_compile(fn,**opts):return Kernel(fn,opts)


def fake_jit(fn=None,**opts):
    def decorate(f):
        @functools.wraps(f)
        def wrapped(*args,**kwargs):return fake_compile(f(*args,**kwargs),**opts)
        wrapped.__jit_impl__=types.SimpleNamespace(out_idx=opts.get('out_idx'),workspace_idx=opts.get('workspace_idx'),pass_configs=opts.get('pass_configs'))
        return wrapped
    return decorate(fn) if fn else decorate


class Attrs:
    def __getattr__(self,name):return name


@functools.lru_cache(None)
def source_ast(filename):
    return ast.parse(Path(filename).read_text())


def install_stubs():
    assert not any(n in sys.modules for n in ('torch','tilelang','torch_npu','acl'))
    torch=types.ModuleType('torch'); torch.__file__='<CPU metadata stub>'
    for d in ('float16','bfloat16','float32','float64','int8','uint8','int16','int32','int64','bool'):
        setattr(torch,d,DType(d))
    torch.float=torch.float32;torch.half=torch.float16;torch.long=torch.int64
    torch.Tensor=Tensor;torch.dtype=DType
    def make(*shape,dtype='float32',**kw):return Tensor(_shape(shape),dtype)
    torch.zeros=torch.ones=torch.empty=make
    torch.zeros_like=torch.ones_like=torch.empty_like=lambda t,**kw:Tensor(t.shape,kw.get('dtype',t.dtype))
    torch.full=lambda shape,value,**kw:make(shape,**kw)
    torch.eye=lambda n,**kw:make(n,n,**kw)
    def arange(*args,**kw):
        vals=list(range(*args));return Tensor((len(vals),),kw.get('dtype','int64'),values=vals)
    torch.arange=arange
    def tensor(data,**kw):
        shape=[];p=data
        while isinstance(p,(tuple,list)):
            shape.append(len(p));p=p[0] if p else 0
        return Tensor(shape,kw.get('dtype','float32'),values=data)
    torch.tensor=tensor
    def cat(ts,dim=0):
        shape=list(ts[0].shape);shape[dim]=sum(t.shape[dim] for t in ts)
        for t in ts:assert all(t.shape[i]==shape[i] for i in range(t.ndim) if i!=dim%t.ndim)
        return Tensor(shape,ts[0].dtype)
    torch.cat=cat
    torch.stack=lambda ts,dim=0:cat([t.unsqueeze(dim) for t in ts],dim)
    torch.nn=types.ModuleType('torch.nn');torch.nn.Module=type('Module',(),{})
    F=types.ModuleType('torch.nn.functional');torch.nn.functional=F
    def pad(t,padding,**kw):
        shape=list(t.shape)
        for i in range(0,len(padding),2):shape[-i//2-1]+=padding[i]+padding[i+1]
        return Tensor(shape,t.dtype)
    F.pad=pad
    tl=types.ModuleType('tilelang');tl.__file__=str(native.VENDOR/'tilelang/__init__.py')
    T=types.ModuleType('tilelang.language')
    T.symbolic=Expr;T.Tensor=Spec;T.prim_func=lambda f:f
    T.ceildiv=lambda a,b:(a+b-1)//b
    T.reduce_sum=T.reduce_max=T.reduce_min=lambda *args,**kw:None
    T.macro=lambda f:f
    T.tile=types.SimpleNamespace(**{n:(lambda *args,**kw:None) for n in ('round','sigmoid','silu','leaky_relu')})
    tl.math=types.SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length())
    tl.cdiv=lambda a,b:(a+b-1)//b
    tl.language=T;tl.PassConfigKey=Attrs();tl.jit=fake_jit;tl.compile=fake_compile
    intr=types.ModuleType('tilelang.intrinsics')
    intr.make_zn_layout=intr.make_nz_layout=lambda *args,**kw:None
    api=types.ModuleType('benchmarks.api')
    from dataclasses import make_dataclass
    api.Implementation=make_dataclass('Implementation',[('run',object),('args',object,None),('reset',object,None),('noncomparable_reason',object,None)])
    for name,module in [('torch',torch),('torch.nn',torch.nn),('torch.nn.functional',F),('tilelang',tl),('tilelang.language',T),('tilelang.intrinsics',intr),('benchmarks.api',api)]:
        sys.modules[name]=module
    native.selected_tree=functools.lru_cache(None)(native.selected_tree)
    original=native.load
    def load(*args):
        LOADS.append(args)
        return original(*args)
    native.load=load


def main():
    inventory=json.loads((HERE/'inventory_snapshot.json').read_text())
    assert len(inventory['operators'])==205
    assert set(PROVIDERS)<={op['op'] for op in inventory['operators']}
    for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    install_stubs()
    import yaml
    specs={};manifests={}
    for row in inventory['operators']:
        path=row['manifest']
        if path not in manifests:manifests[path]=yaml.safe_load((native.WORKSPACE/'TileOPs'/path).read_text())
        specs[row['op']]=manifests[path][row['op']]['signature']
    failures=[];passed=[];run_pass=[];case_records=[]
    native.definition_records=functools.lru_cache(None)(native.definition_records)
    for row in inventory['cases']:
        p,_=parameters(row);op=row['op']
        if not assess(op,p).supported:continue
        # Factory construction exercises source extraction and actual original
        # Python builder signatures. Inputs are shape-only objects, never arrays.
        t={key:Tensor(sh,dt,(key,)) for key,(sh,dt) in row['tensors'].items()}
        case=types.SimpleNamespace(id=row['case_id'],params=p,arguments={},inputs=tuple(t[k] for k in ('a','b') if k in t))
        if 'rescale_factors' in t:case.arguments['rescale_factors']=t['rescale_factors']
        for key in ('cu_seqlens_q','cu_seqlens_kv','cu_seqlens'):
            if key not in t:continue
            count=t[key].shape[0]-1
            total=(t['q'].shape[0] if key=='cu_seqlens_q' else t['k'].shape[0] if key=='cu_seqlens_kv' else t['x'].shape[1])
            t[key].values=[i*total//count for i in range(count+1)]
        if op=='GQAPagedFwdOp':t['cache_seqlens'].values=[t['page_table'].shape[1]*t['k_pages'].shape[1]]*t['cache_seqlens'].shape[0]
        if op=='MHADecodePagedWithKVCacheFwdOp':t['real_seqlen_kv'].values=[t['block_table'].shape[1]*p['page_size']]*t['real_seqlen_kv'].shape[0]
        if op=='MoEGroupedGemmFwdOp':
            count=t['layout_metadata'].shape[0]
            if len(t['a'].shape)==3:t['layout_metadata'].values=[t['a'].shape[1]]*count
            else:t['layout_metadata'].values=[(i+1)*t['a'].shape[0]//count for i in range(count)]
        if 'batch_sizes' in t:
            count=t['batch_sizes'].shape[0];total=t['a'].shape[0]
            offsets=[i*total//count for i in range(count+1)]
            t['batch_sizes'].values=[b-a for a,b in zip(offsets[:-1],offsets[1:])]
            t['batch_offsets'].values=offsets[:-1]
        inputs=tuple(t.get(key) for key in specs[op]['inputs'])
        case.inputs=inputs
        phase='factory'
        first_load=len(LOADS);first_call=len(CALLS)
        try:
            candidate=PROVIDERS[op](case)
            assert candidate.api.startswith('tilelang-ascend::')
            assert candidate.source_sha256
            passed.append((op,row['case_id']))
            phase='run_metadata'
            output=candidate.implementation.run(*inputs)
            expected=[t[key] for key in specs[op]['outputs']]
            if op.startswith('CausalConv1d'):expected.append(t['conv_state'])
            got=list(output) if isinstance(output,(tuple,list)) else [output]
            assert len(got)==len(expected),(len(got),len(expected))
            for actual,wanted in zip(got,expected):
                assert actual.shape==wanted.shape,(actual.shape,wanted.shape)
                assert actual.dtype==wanted.dtype,(actual.dtype,wanted.dtype)
            run_pass.append((op,row['case_id']))
            sources=[]
            for args in sorted(set(LOADS[first_load:])):
                sources.extend(native.definition_records(args[0],args[1:]))
            case_records.append({'op':op,'case_id':row['case_id'],'api':candidate.api,
                'adaptation':candidate.adaptation,'timing_mode':candidate.timing_mode,
                'source_sha256':candidate.source_sha256,'sources':sources,
                'outputs':[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in got],
                'input_write_protection':'metadata provenance plus original AST destination audit',
                'native_metadata_calls':len(CALLS)-first_call,'numerical_receipt':None})
        except Exception as exc:
            failures.append({'op':op,'case_id':row['case_id'],'phase':phase,'error':f'{type(exc).__name__}: {exc}'})
    records=[]
    for args in sorted(set(LOADS)):
        tree,path=native.selected_tree(args[0],args[1:])
        # Every selected AST node must be byte-for-byte structurally identical
        # to its matching original source node, including its body and schedule.
        original=source_ast(str(path))

        for node in tree.body:
            assert any(ast.dump(n,include_attributes=False)==ast.dump(node,include_attributes=False)
                       for n in ast.walk(original) if getattr(n,'lineno',None)==node.lineno)
        records.append({'file':str(path.relative_to(native.WORKSPACE)),'requested':list(args[1:]),
                        'selected_nodes':len(tree.body),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'schema':'tla-cpu-static/v1','numerical_validation':False,'real_runtime_imports':False,
            'device_runs':0,'providers':len(PROVIDERS),'factory_metadata_pass':len(passed),
            'run_metadata_pass':len(run_pass),'metadata_failures':len(failures),'failures':failures,'source_loads':records,'native_metadata_calls':len(CALLS)}
    (HERE/'static_case_receipts.json').write_text(json.dumps({'schema':'tla-static-case/v1',
        'warning':'Synthetic metadata only; no actual tensors, native kernel execution, numerical validation or device evidence.',
        'cases':case_records},indent=2)+'\n')
    (HERE/'static_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_loads','failures')}))
    for failure in failures[:30]:print(failure)
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
