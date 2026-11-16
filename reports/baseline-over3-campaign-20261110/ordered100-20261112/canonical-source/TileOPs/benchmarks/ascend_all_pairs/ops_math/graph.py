"""Descriptor-only bridge to unchanged ops-math graph kernels."""
import ctypes as C
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from .specs import SPECS, SOURCE, VENDOR, files, REDUCTIONS


@lru_cache(maxsize=1)
def deployment_hashes():
    from benchmarks.ascend_baselines import source_hashes
    deployed=[q for q in VENDOR.rglob('*') if q.is_file() and q.suffix in ('.so','.json','.py','.cpp','.h')]
    return source_hashes(*deployed)

def make(case, op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    spec=SPECS[op]; graph=spec['graph']; p=case.params
    route_sources=[]
    weighted=bool(p.get('_weighted'))
    argvalue=bool(p.get('_argvalue'));cdist=bool(p.get('_cdist'));topk=p.get('_topk')
    for selected,folder,name in ((argvalue,'arg_max_with_value','ArgMaxWithValue'),(cdist,'cdist','Cdist'),(topk is not None,'top_k_v2','TopKV2')):
        if selected:
            graph=name
            route_sources += [q for part in ('op_host','op_kernel','op_graph') for q in (SOURCE/'math'/folder/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h','.py')]
    if weighted:
        graph='ReduceMeanWithCount'
        route_sources += [q for part in ('op_host','op_kernel','op_graph') for q in (SOURCE/'math/reduce_mean_with_count'/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h','.py')]
    if op=='LogicalNotFwdOp' and p.get('input',(None,'bool'))[1]!='bool':
        graph='Equal';route_sources+=files('EqFwdOp')
    if op=='LogicalAndFwdOp' and p.get('input',(None,'bool'))[1]!='bool':
        graph='ReduceAll';route_sources+=files('AllFwdOp')
    configured={Path(s).resolve() for s in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if s}
    if VENDOR.resolve() not in configured:
        raise RuntimeError('Configure isolated allpairs_math_math vendor before runtime import.')
    config=VENDOR/'op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'
    import json
    if not config.exists() or graph not in json.loads(config.read_text()):
        raise RuntimeError(f'Original {graph} graph is not deployed in isolated vendor.')
    if op in ('AddFwdOp','SubFwdOp') and p.get('alpha',1)!=1:
        raise UnsupportedCase('Native graph Sub has no alpha parameter; alpha != 1 requires arithmetic adaptation.')
    if op=='DivFwdOp' and p.get('rounding_mode') is not None:
        graph={'floor':'FloorDiv','trunc':'TruncateDiv'}[p['rounding_mode']]
        if graph not in json.loads(config.read_text()):
            raise RuntimeError(f'Original rounding API {graph} not deployed.')
    codes={torch.float32:0,torch.float16:1,torch.int8:2,torch.int32:3,torch.uint8:4,torch.int16:6,torch.int64:9,torch.float64:11,torch.bool:12,torch.complex64:16,torch.complex128:17,torch.bfloat16:27}
    base=C.CDLL('libascendcl.so'); compiler=C.CDLL('libacl_op_compiler.so')
    ptrs=C.POINTER(C.c_void_p)
    def bind(lib,name,args,result=C.c_int):
        f=getattr(lib,name); f.argtypes=args; f.restype=result; return f
    desc=bind(base,'aclCreateTensorDesc',[C.c_int,C.c_int,C.POINTER(C.c_int64),C.c_int],C.c_void_p)
    free_desc=bind(base,'aclDestroyTensorDesc',[C.c_void_p],None)
    const=bind(base,'aclSetTensorConst',[C.c_void_p,C.c_void_p,C.c_size_t])
    buf=bind(base,'aclCreateDataBuffer',[C.c_void_p,C.c_size_t],C.c_void_p)
    free_buf=bind(base,'aclDestroyDataBuffer',[C.c_void_p])
    attr=bind(base,'aclopCreateAttr',[],C.c_void_p)
    free_attr=bind(base,'aclopDestroyAttr',[C.c_void_p],None)
    set_bool=bind(base,'aclopSetAttrBool',[C.c_void_p,C.c_char_p,C.c_uint8])
    set_int=bind(base,'aclopSetAttrInt',[C.c_void_p,C.c_char_p,C.c_int64])
    set_type=bind(base,'aclopSetAttrDataType',[C.c_void_p,C.c_char_p,C.c_int])
    set_list=bind(base,'aclopSetAttrListInt',[C.c_void_p,C.c_char_p,C.c_int,C.POINTER(C.c_int64)])
    set_float=bind(base,'aclopSetAttrFloat',[C.c_void_p,C.c_char_p,C.c_float])
    option=bind(compiler,'aclSetCompileopt',[C.c_int,C.c_char_p])
    execute=bind(compiler,'aclopCompileAndExecute',[C.c_char_p,C.c_int,ptrs,ptrs,C.c_int,ptrs,ptrs,C.c_void_p,C.c_int,C.c_int,C.c_char_p,C.c_void_p])
    def check(rc,name):
        if rc: raise RuntimeError(f'{name} returned {rc}')
    check(option(10,b'enable'),'aclSetCompileopt(JIT)')
    pending=[]
    out_spec=p.get('output')
    if out_spec is None:
        out_spec=p.get('indices',p.get('y',p.get('var')))
    if out_spec is None: raise UnsupportedCase('No single output tensor specification in manifest.')
    out_shape,out_dtype=out_spec
    def run(*args):
        xs=[x.contiguous() if isinstance(x,torch.Tensor) else x for x in args]
        constants={}; attrs={'axes':[-1],'keep_dims':False} if weighted else {}; x=next(t for t in xs if isinstance(t,torch.Tensor))
        # The original bitwise graph supports integer storage, not BOOL. Boolean
        # values widen exactly to {0,1}; AND/OR/XOR preserve that set, so the
        # original BOOL result is restored without changing the calculation.
        bool_bitwise=op in ('BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp') and x.dtype==torch.bool
        if bool_bitwise:xs=[t.to(torch.int8) for t in xs]
        if argvalue:attrs.update(dimension=-1,keep_dims=False,indice_dtype=9)
        if cdist:attrs['p']=float(p['_p'])
        if topk is not None:
            xs.append(torch.tensor(topk,device=x.device,dtype=torch.int64));constants[len(xs)-1]=[topk]
            attrs.update(sorted=True,dim=-1,largest=True,indices_dtype=9)
        if op=='LogicalNotFwdOp' and graph=='Equal':xs.append(torch.zeros((),device=x.device,dtype=x.dtype))
        if op=='ReluFwdOp':xs.append(torch.zeros((),device=x.device,dtype=x.dtype))
        if op=='LogicalAndFwdOp' and graph=='ReduceAll':
            xs=[torch.stack([v.expand(out_shape) for v in xs],dim=-1),torch.tensor([len(out_shape)],device=x.device,dtype=torch.int64)]
            constants[1]=[len(out_shape)];attrs['keep_dims']=False
        if op=='AnyFwdOp' and x.dtype in (torch.float16,torch.bfloat16): xs[0]=xs[0].float()
        if (op in REDUCTIONS and not weighted and not argvalue) or op in ('ArgmaxFwdOp','ArgminFwdOp','CumsumFwdOp'):
            d=p.get('dim',-1)
            if d is None: axes=list(range(x.ndim))
            elif isinstance(d,(list,tuple)): axes=[a%x.ndim for a in d]
            else: axes=[d%x.ndim] if x.ndim else [0]
            if op in ('ArgmaxFwdOp','ArgminFwdOp') and d is None:
                xs[0]=xs[0].reshape(-1); axes=[0]
            dtype=p.get('dtype')
            if dtype:
                target=getattr(torch,dtype)
                if target!=x.dtype:
                    if x.dtype not in (torch.bfloat16,torch.float16) or target!=torch.float32:
                        raise UnsupportedCase('Requested cast is not an audited lossless widening.')
                    xs[0]=xs[0].to(target)
            axis=torch.tensor(axes,device=x.device,dtype=torch.int64)
            if op in ('ArgmaxFwdOp','ArgminFwdOp','CumsumFwdOp'): axis=axis.reshape(())
            xs.append(axis); constants[len(xs)-1]=axes
            if op in REDUCTIONS: attrs['keep_dims']=bool(p.get('keepdim',False))
            if op in ('ArgmaxFwdOp','ArgminFwdOp'): attrs['dtype']=9
        if op in ('VarFwdOp','VarMeanFwdOp','StdFwdOp'):
            dim=p.get('dim',-1)
            attrs.update(dim=list(range(x.ndim)) if dim is None else list(dim) if isinstance(dim,(tuple,list)) else [dim],correction=int(p.get('correction',1)),keepdim=bool(p.get('keepdim',False)),is_mean_out=True)
        if op=='RoundFwdOp': attrs['decimals']=int(p.get('decimals',0))
        if op=='LerpScalarFwdOp': xs.append(torch.tensor(p.get('weight',0.5),device=x.device,dtype=x.dtype))
        if op in ('ClampScalarFwdOp','HardtanhFwdOp'):
            lo=p.get('min',p.get('min_val')); hi=p.get('max',p.get('max_val'))
            lo=float('-inf') if lo is None else lo; hi=float('inf') if hi is None else hi
            xs.extend(torch.tensor(v,device=x.device,dtype=x.dtype) for v in (lo,hi))
        if op=='ClampTensorFwdOp':
            if xs[1] is None: xs[1]=torch.tensor(float('-inf'),device=x.device,dtype=x.dtype)
            if xs[2] is None: xs[2]=torch.tensor(float('inf'),device=x.device,dtype=x.dtype)
        if op=='MaskedFillScalarFwdOp': xs.append(torch.tensor(p['value'],device=x.device,dtype=x.dtype))
        if op=='NanToNumFwdOp':
            attrs.update(nan=float(p.get('nan',0.)),posinf=float(p.get('posinf') if p.get('posinf') is not None else torch.finfo(x.dtype).max),neginf=float(p.get('neginf') if p.get('neginf') is not None else torch.finfo(x.dtype).min))
        # Arg graph removes the reduced dimension; keepdim is a view-only adaptation.
        shape=out_shape
        if op in ('ArgmaxFwdOp','ArgminFwdOp') and p.get('keepdim'):
            shape=tuple(v for i,v in enumerate(x.shape) if i not in axes)
        output=torch.empty(shape,device=x.device,dtype=torch.int8 if bool_bitwise else getattr(torch,out_dtype))
        outputs=[output]
        if argvalue:outputs=[torch.empty(shape,device=x.device,dtype=torch.int64),output]
        if topk is not None:outputs=[output,torch.empty(shape,device=x.device,dtype=torch.int64)]
        if op in ('VarFwdOp','VarMeanFwdOp','StdFwdOp'): outputs.append(torch.empty_like(output))
        rec={'d':[],'b':[],'keep':[xs,outputs],'a':None}; pending.append(rec)
        for i,t in enumerate(xs+outputs):
            dims=(C.c_int64*t.ndim)(*t.shape); dp=desc(codes[t.dtype],t.ndim,dims,2); bp=buf(t.data_ptr(),t.numel()*t.element_size())
            rec['d'].append(dp);rec['b'].append(bp);rec['keep'].append(dims)
            if not dp or not bp: raise RuntimeError('Graph descriptor allocation failed')
            if i in constants:
                values=constants[i]; data=(C.c_int64*len(values))(*values);rec['keep'].append(data)
                check(const(dp,C.cast(data,C.c_void_p),C.sizeof(data)),'aclSetTensorConst')
        a=attr();rec['a']=a
        if not a: raise RuntimeError('Attribute allocation failed')
        for k,v in attrs.items():
            if isinstance(v,list):
                data=(C.c_int64*len(v))(*v);rec['keep'].append(data)
                check(set_list(a,k.encode(),len(v),data),'aclopSetAttrListInt:'+k);continue
            setter=set_type if k in ('dtype','indice_dtype') else set_bool if isinstance(v,bool) else set_int if isinstance(v,int) else set_float
            check(setter(a,k.encode(),v),'aclopSetAttr:'+k)
        n=len(xs); no=len(outputs); ids=(C.c_void_p*n)(*rec['d'][:n]);ibs=(C.c_void_p*n)(*rec['b'][:n]);ods=(C.c_void_p*no)(*rec['d'][n:]);obs=(C.c_void_p*no)(*rec['b'][n:]);rec['keep'].extend((ids,ibs,ods,obs))
        call_start=datetime.now(timezone.utc).isoformat()
        check(execute(graph.encode(),n,ids,ibs,no,ods,obs,a,0,0,None,torch.npu.current_stream().npu_stream),f'aclopCompileAndExecute({graph})')
        trace=os.environ.get('OPS_MATH_GRAPH_CALLS')
        if trace:
            with open(trace,'a') as stream:
                stream.write(json.dumps({'graph':graph,'started_at':call_start,'finished_at':datetime.now(timezone.utc).isoformat(),'inputs':[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in xs],'outputs':[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in outputs]})+'\n')
        if bool_bitwise:return output.reshape(out_shape).to(torch.bool)
        return tuple(t.reshape(out_shape) for t in outputs) if op=='VarMeanFwdOp' or argvalue or topk is not None else output.reshape(out_shape)
    def cleanup():
        torch.npu.synchronize()
        while pending:
            r=pending.pop()
            for d in r['d']:
                if d: free_desc(d)
            for b in r['b']:
                if b: check(free_buf(b),'aclDestroyDataBuffer')
            if r['a']: free_attr(r['a'])
    return Candidate(Implementation(run),f'ops-math::{graph} via aclopCompileAndExecute',
        'Original graph and schedule; per-call contiguous packing, scalar/axis descriptors, optional exact widening and output view. No cached input data.',
        source_hashes(Path(__file__),Path(__file__).with_name('specs.py'),*files(op),*route_sources) | deployment_hashes(),
        limitations=('Acceptance requires own-vendor compilation and native launch evidence.',),cleanup=cleanup)
