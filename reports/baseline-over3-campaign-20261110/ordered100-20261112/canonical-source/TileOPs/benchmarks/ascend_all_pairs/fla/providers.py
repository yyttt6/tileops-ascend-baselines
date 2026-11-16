"""Layout/API adapters only. Device arithmetic belongs to the original FLA source."""
from pathlib import Path


def _triton_module(stem):
    import importlib.util
    import sys
    from benchmarks.ascend_baselines import LIBRARIES
    source = LIBRARIES / f"flash-linear-attention-npu/fla/ops/triton/triton_core/{stem}.py"
    name = f"_allpairs_fla_{stem}"
    if name in sys.modules:
        return sys.modules[name], source
    spec = importlib.util.spec_from_file_location(name, source)
    if stem == "l2norm":
        import types
        package_name = "_allpairs_fla_core"
        if package_name not in sys.modules:
            package=types.ModuleType(package_name)
            package.__path__=[str(source.parent)]
            sys.modules[package_name]=package
        name=package_name+".l2norm"
        spec=importlib.util.spec_from_file_location(name,source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if stem == "causal_conv1d":
        from triton.language.extra.cann.extension import vec_ops
        if not hasattr(module.tl, "extract_slice"):
            module.tl.extract_slice = vec_ops.extract_slice
    return module, source


def rmsnorm(case):
    import math
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    norm,norm_source=_triton_module("l2norm")
    conv,conv_source=_triton_module("causal_conv1d")
    fused=case._call.signature.name=="FusedAddRMSNormFwdOp"
    p=case.params
    def native_product(x,w,residual=None):
        shape=x.shape
        n=x.numel()
        channels=n+-n%256
        data=F.pad(x.contiguous().reshape(-1),(0,channels-n)).reshape(1,1,channels)
        weights=F.pad(w.expand(shape).contiguous().reshape(-1),(0,channels-n)).reshape(1,channels)
        res=None if residual is None else F.pad(residual.contiguous().reshape(-1),(0,channels-n)).reshape(1,1,channels)
        out,_=conv.causal_conv1d_fwd_impl_old(data,weights,bias=None,residual=res)
        return out.reshape(-1)[:n].reshape(shape)

    def run(x,*args):
        if fused:
            residual,weight=args
            x=native_product(x,torch.ones((),device=x.device,dtype=x.dtype),residual=residual)
        else:
            weight=args[0]
        shape=tuple(p.get("normalized_shape",(x.shape[-1],)))
        dim=math.prod(shape)
        data=x.contiguous().reshape(-1,dim)
        epsilon=(torch.finfo(torch.float32).eps if p.get("eps") is None else p["eps"])*dim
        if dim<=65536//x.element_size():
            normal,_=norm.l2norm_fwd(data,eps=epsilon,output_dtype=torch.float32)
        else:
            lengths=[]
            for start in range(0,dim,8192):
                _,inverse=norm.l2norm_fwd(data[:,start:start+8192].contiguous(),eps=0.0,output_dtype=torch.float32)
                _,length=norm.l2norm_fwd(inverse.reshape(-1,1),eps=0.0,output_dtype=torch.float32)
                lengths.append(length)
            _,inverse=norm.l2norm_fwd(torch.stack(lengths,dim=1),eps=epsilon,output_dtype=torch.float32)
            normal=native_product(data.float(),inverse[:,None])
        normal=normal.reshape(x.shape)
        normal=native_product(normal,torch.full((),math.sqrt(dim),device=x.device,dtype=torch.float32))
        if weight is not None:
            normal=native_product(normal,weight.float())
        result=normal.to(x.dtype)
        return (result,x) if fused else result

    return Candidate(Implementation(run),"FLA l2norm_fwd + causal_conv1d_fwd_impl_old(W=1)",
        "Flatten normalized dimensions; original L2 epsilon=N*eps and FP32 result; dimensions exceeding one native L2 block compose native chunk rstd, scalar reciprocal L2, then a native global L2 rstd; original single-tap convolutions apply sqrt(N), affine weight, and optional residual. Only input/layout/cast/API composition; schedules untouched.",
        source_hashes(Path(__file__),norm_source,norm_source.with_name("utils.py"),conv_source))


def cumsum(case):
    import importlib.util
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, source_hashes

    source = LIBRARIES / "flash-linear-attention-npu/fla/ops/triton/triton_core/cumsum.py"
    spec = importlib.util.spec_from_file_location("_allpairs_fla_cumsum", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dim = case.params.get("dim", -1)
    hashes=source_hashes(Path(__file__),source)
    affine=None
    if case.inputs[0].shape[dim]>16384:
        from .native_affine import make_affine
        affine,affine_hashes=make_affine()
        hashes.update(affine_hashes)

    def bounded_prefix(data):
        rows,size=data.shape
        if size<=16384:
            extent=1<<(size-1).bit_length()
            packed=F.pad(data,(0,extent-size)).reshape(-1,extent,1)
            return module.chunk_local_cumsum(packed,chunk_size=extent,output_dtype=torch.float32)[...,0][:,:size]
        chunks=(size+4095)//4096
        packed=F.pad(data,(0,chunks*4096-size)).reshape(-1,4096,1)
        local=module.chunk_local_cumsum(packed,chunk_size=4096,output_dtype=torch.float32)[...,0].reshape(rows,chunks,4096)
        totals=bounded_prefix(local[:,:,-1].contiguous())
        offsets=F.pad(totals[:,:-1],(1,0)).unsqueeze(-1)
        return affine(local,1.0,offsets).reshape(rows,-1)[:,:size]

    def run(x):
        data = x.movedim(dim, -1).contiguous()
        size = data.shape[-1]
        if affine is not None:
            return bounded_prefix(data.reshape(-1,size)).to(x.dtype).reshape(data.shape).movedim(-1,dim).contiguous()
        extent = 1 << (size - 1).bit_length()
        padded = F.pad(data, (0, extent - size)).reshape(-1, extent, 1)
        out = module.chunk_local_cumsum(padded, chunk_size=extent, output_dtype=x.dtype)
        return out[..., 0][:, :size].reshape(data.shape).movedim(-1, dim).contiguous()

    return Candidate(Implementation(run), "fla.ops.triton.triton_core.cumsum.chunk_local_cumsum",
                     "Move cumulative axis last; zero-pad native chunks and crop. Wide rows use original FP32 chunk prefixes, original prefixes of chunk totals, and original tap/residual addition of exclusive chunk offsets, with a final manifest cast. Original host/kernel/schedule unchanged.",hashes)


def reduce(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    module, source = _triton_module("cumsum")
    p = case.params
    dim = p.get("dim", -1)
    keepdim = p.get("keepdim", False)
    mean = case._call.signature.name == "MeanFwdOp"

    def run(x):
        dims = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
        dims = tuple(d % x.ndim for d in dims)
        left = tuple(d for d in range(x.ndim) if d not in dims)
        shape = tuple(x.shape[d] for d in left)
        width = 1
        for d in dims:
            width *= x.shape[d]
        data = x.permute(left + dims).contiguous().reshape(-1, width)
        parts=width
        while parts>16384:
            chunks=(parts+4095)//4096
            padded=F.pad(data,(0,chunks*4096-parts)).reshape(-1,4096,1)
            data=module.chunk_local_cumsum(padded,chunk_size=4096,output_dtype=torch.float32)[:,-1,0].reshape(data.shape[0],chunks)
            parts=chunks
        extent = 1 << (parts - 1).bit_length()
        padded = F.pad(data, (0, extent - parts)).reshape(-1, extent, 1)
        output_dtype = getattr(torch, p["dtype"]) if p.get("dtype") else x.dtype
        value = module.chunk_local_cumsum(padded, chunk_size=extent, scale=(1.0 / width if mean else None), output_dtype=output_dtype)[:, parts - 1, 0]
        if keepdim:
            shape = tuple(1 if d in dims else x.shape[d] for d in range(x.ndim))
        return value.reshape(shape)

    return Candidate(Implementation(run), "FLA chunk_local_cumsum(last prefix)",
                     "Permute reduction axes last, zero-pad to power of two, select final prefix; long rows compose bounded original scans with FP32 intermediate sums. Mean applies original API scale=1/N once to the complete reduction.",
                     source_hashes(Path(__file__), source))


def pointwise(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    module, source = _triton_module("causal_conv1d")
    name = case._call.signature.name
    alpha = case.params.get("alpha", 1)

    def run(x, y=None):
        if name == "SiluAndMulFwdOp":
            x, y = x.chunk(2, dim=-1)
        shape = tuple(x.shape) if y is None else tuple(torch.broadcast_shapes(x.shape, y.shape))
        xp = x.expand(shape).contiguous().reshape(-1)
        n = xp.numel()
        extent = n + -n % 256
        data = F.pad(xp, (0, extent - n)).reshape(1, 1, extent)
        residual = None
        activation = "silu" if name in ("SiluFwdOp", "SiluAndMulFwdOp") else None
        if name in ("MulFwdOp",):
            weights = F.pad(y.expand(shape).contiguous().reshape(-1), (0, extent - n)).reshape(1, extent)
        else:
            value = -1.0 if name == "NegFwdOp" else 1.0
            weights = torch.full((1, extent), value, device=x.device, dtype=x.dtype)
        if name in ("AddFwdOp", "SubFwdOp"):
            # Bias/residual are native arithmetic; no torch arithmetic on data.
            yp = F.pad(y.expand(shape).contiguous().reshape(-1), (0, extent - n)).reshape(1, 1, extent)
            residual = data
            data = yp
            weights.fill_(alpha if name == "AddFwdOp" else -alpha)
        out, _ = module.causal_conv1d_fwd_impl_old(data, weights, bias=None, residual=residual, activation=activation)
        if name == "SiluAndMulFwdOp":
            weights = F.pad(y.expand(shape).contiguous().reshape(-1), (0, extent - n)).reshape(1, extent)
            out, _ = module.causal_conv1d_fwd_impl_old(out, weights, bias=None, residual=None)
        return out.reshape(-1)[:n].reshape(shape)

    return Candidate(Implementation(run), "FLA causal_conv1d_fwd_impl_old(W=1)",
                     "Broadcast/flatten/pad input into channels; original single tap product, residual add, or SiLU activation. No host/kernel tile or schedule changes; current extract_slice API alias only.",
                     source_hashes(Path(__file__), source))



def select_deployment(symbol):
    """Select an audited original package before importing its lazy public API."""
    import os
    import sys
    from benchmarks.ascend_baselines import WORKSPACE
    if os.environ.get('ASCEND_CLASSIC_FLA_PACKAGE'):
        return Path(os.environ['ASCEND_CLASSIC_FLA_PACKAGE']).resolve()
    if symbol=='aclnnCausalConv1d':
        deployed=WORKSPACE/'.venv/lib/python3.12/site-packages/fla_npu'
    elif symbol=='aclnnChunkKdaFwd' and (WORKSPACE/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla/kda_native_ready.json').is_file():
        deployed=WORKSPACE/'.cache/all-pairs/fla/python_binary_kda/fla_npu'
    elif symbol in ('aclnnChunkGatedDeltaRuleFwdPrepare','aclnnChunkGatedDeltaRuleBwd','aclnnKdaGateCumsum') and (WORKSPACE/'.cache/all-pairs/fla/python_binary_partial/fla_npu').is_dir():
        deployed=WORKSPACE/'.cache/all-pairs/fla/python_binary_partial/fla_npu'
    elif symbol in ('aclnnChunkKdaFwd','aclnnChunkGatedDeltaRuleFwdPrepare','aclnnChunkGatedDeltaRuleBwd','aclnnKdaGateCumsum'):
        deployed=WORKSPACE/'.cache/all-pairs/fla/python_binary/fla_npu'
    else:
        suffix='flash-linear-attention-npu'+('-recurrent-kda' if symbol=='aclnnRecurrentKda' else '')
        deployed=WORKSPACE/'.cache/ascend-classic/native-pilot-20261020'/suffix/'python_binary/fla_npu'
    if not deployed.is_dir():
        raise RuntimeError('Original native deployment not available: '+str(deployed))
    os.environ['ASCEND_CLASSIC_FLA_PACKAGE']=str(deployed)
    sys.path.insert(0,str(deployed.parent))
    vendor=deployed/'opp/vendors/fla_npu_transformer'
    os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)+os.pathsep+str(deployed/'opp')
    os.environ['LD_LIBRARY_PATH']=str(vendor/'op_api/lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    return deployed.resolve()


def _native(symbol):
    import importlib
    import os
    from benchmarks.ascend_baselines import LIBRARIES, WORKSPACE, source_hashes
    from benchmarks.ascend_extra_baselines import _owned_native
    source = LIBRARIES / "flash-linear-attention-npu"
    deployed = select_deployment(symbol)
    original = source / "torch_custom/fla_npu/fla_npu"
    files = sorted(original.rglob("*.py"))
    for path in files:
        target = deployed / path.relative_to(original)
        if not target.exists() or target.read_bytes() != path.read_bytes():
            raise RuntimeError(f"FLA deployed Python/source mismatch: {target}")
    binary = deployed / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    _owned_native(binary, [symbol, symbol + "GetWorkspaceSize"])
    os.environ["FLA_NPU_STABLE_ABI"] = "ctypes"
    public = importlib.import_module("fla_npu.ops.ascendc")
    if Path(public.__file__).resolve() != deployed / "ops/ascendc/__init__.py":
        raise RuntimeError("Wrong FLA package; select audited deployment before runtime initialization.")
    folder = {"aclnnChunkGatedDeltaRuleFwdPrepare": "gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare", "aclnnRecurrentKda": "kda/recurrent_kda", "aclnnChunkGatedDeltaRuleFwd": "gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd", "aclnnCausalConv1d": "gdn/gdn_preprocess/causal_conv1d", "aclnnChunkKdaFwd":"kda", "aclnnKdaGateCumsum":"kda/kda_gate_cumsum", "aclnnChunkGatedDeltaRuleBwd":"gdn/chunk_gdn_bwd", "aclnnChunkFwdO":"gdn/chunk_gdn_fwd/chunk_fwd_o", "aclnnChunkFwdH":"gdn/chunk_gdn_fwd/chunk_fwd_h"}[symbol]
    files += sorted(p for p in (source / "fla/ops/ascendc" / folder).rglob("*") if p.suffix in (".cpp", ".h", ".hpp"))
    files += sorted(p for p in (deployed / "opp").rglob("*") if p.suffix in (".o", ".so", ".json"))
    return public, source_hashes(Path(__file__), binary, *files)


def bmm(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    a,b=case.inputs
    if a.shape[-1]>128:
        from .spatial import gemm
        return gemm(case)
    public,hashes=_native("aclnnChunkFwdO")
    def run(a,b):
        batch,m,k=a.shape
        n=b.shape[-1]
        q=F.pad(a,(0,128-k)).unsqueeze(1).contiguous()
        zero_k=torch.zeros_like(q)
        zero_v=torch.zeros((batch,1,m,128),device=a.device,dtype=a.dtype)
        gate=torch.zeros((batch,1,m),device=a.device,dtype=torch.float32)
        pieces=[]
        for first in range(0,n,128):
            last=min(n,first+128)
            state=F.pad(b[:,:,first:last],(0,128-(last-first),0,128-k))
            state=state[:,None,None].expand(batch,(m+63)//64,1,128,128).contiguous()
            output=public.npu_chunk_fwd_o(q,zero_k,zero_v,state,scale=1.0,g=gate,
                    chunk_size=64,output_layout="BNSD",use_exp2=False)
            pieces.append(output[:,0,:,:last-first])
        if public.BACKENDS.get("npu_chunk_fwd_o")!="ctypes":raise RuntimeError("Wrong original ChunkFwdO backend")
        return torch.cat(pieces,dim=-1).contiguous()
    return Candidate(Implementation(run),"fla_npu.ops.ascendc.npu_chunk_fwd_o(k=v=g=0,scale=1)",
        "Native output reduces to q@state. Pad K<=128 and output feature blocks to128, repeat original B as each chunk's state, crop/concatenate N. No split-K rounding or torch arithmetic.",hashes)


def chunk_kda(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    public,hashes=_native("aclnnChunkKdaFwd")
    p=case.params
    op=case._call.signature.name
    external_norm=case.inputs[0].dtype==torch.float16 and p.get("use_qk_l2norm_in_kernel",False)
    if external_norm:
        norm,norm_source=_triton_module("l2norm")
        hashes.update(source_hashes(norm_source,norm_source.with_name("utils.py")))

    def run(q,k,v,g,beta,initial=None,offsets=None,offsets_cpu=None,a_log=None,dt_bias=None):
        dk,dv=q.shape[-1],v.shape[-1]
        scale=p.get("scale")
        scale=dk**-0.5 if scale is None else scale
        if op=="GDNFwdOp":
            g=g.unsqueeze(-1).expand(*g.shape,dk).contiguous()
            if dt_bias is not None:dt_bias=dt_bias.reshape(-1,1).expand(-1,dk).contiguous().reshape(-1)
        q,k,v=F.pad(q,(0,128-dk)),F.pad(k,(0,128-dk)),F.pad(v,(0,128-dv))
        g=F.pad(g,(0,128-dk)).float()
        if dt_bias is not None:
            dt_bias=F.pad(dt_bias.reshape(v.shape[2],dk),(0,128-dk)).contiguous().reshape(-1)
        beta=beta.float()
        if external_norm:
            q=norm.l2norm_fwd(q,eps=1e-6)[0]
            k=norm.l2norm_fwd(k,eps=1e-6)[0]
        state_v_first=p.get("state_v_first",False)
        state=None if initial is None else F.pad(initial,(0,128-dk,0,128-dv) if state_v_first else (0,128-dv,0,128-dk)).clone()
        metadata=offsets_cpu if offsets_cpu is not None else offsets
        cu=None if metadata is None else tuple(metadata.cpu().tolist())
        result=public.npu_chunk_kda_fwd(q,k,v,g,beta,scale=scale,layout="BSND",
            initial_state=state,output_final_state=True,cu_seqlens=cu,state_v_first=state_v_first,
            use_qk_l2norm_in_kernel=p.get("use_qk_l2norm_in_kernel",False) and not external_norm,
            use_gate_in_kernel=p.get("use_gate_in_kernel",False),A_log=a_log,dt_bias=dt_bias,
            use_beta_sigmoid_in_kernel=p.get("use_beta_sigmoid_in_kernel",False),
            allow_neg_eigval=p.get("allow_neg_eigval",False),
            safe_gate=p.get("lower_bound") is not None,lower_bound=p.get("lower_bound"))
        if public.BACKENDS.get("npu_chunk_kda_fwd")!="ctypes":raise RuntimeError("Wrong native KDA backend")
        final=result[1]
        return result[0][...,:dv].contiguous(),(final[...,:dv,:dk] if state_v_first else final[...,:dk,:dv]).contiguous()

    return Candidate(Implementation(run),"fla_npu.ops.ascendc.npu_chunk_kda_fwd",
        "Original complete chunk KDA, original raw-gate/beta/L2 options; GDN scalar gate broadcast, feature pad/crop, private state and lossless gate/beta FP32 representation. FP16 uses original FLA L2Norm API when fused L2 is unavailable.",hashes)


def causal(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    public,hashes=_native("aclnnCausalConv1d")
    update=case._call.signature.name=="CausalConv1dUpdateFwdOp"
    activation=case.params.get("activation")

    def run(x,weight,conv_state,*tail):
        state=conv_state.transpose(1,2).contiguous().clone()
        taps=weight.t().contiguous()
        if update:
            bias=tail[0] if tail else None
            import torch
            # The manifest's W-slot history shifts left before the dot product.
            # Supply its last W-1 slots as FN initial history, and explicitly
            # return the exact copy-only shifted W-slot state.
            final=torch.cat((conv_state[...,1:],x.unsqueeze(-1)),dim=-1)
            history=conv_state[...,1:].transpose(1,2).contiguous().clone()
            flags=torch.ones((x.shape[0],),device=x.device,dtype=torch.bool)
            out=public.npu_causal_conv1d_fn(x.unsqueeze(1).contiguous(),taps,bias,history,
                    has_initial_state=flags,activation=activation,null_block_id=None)[:,0]
            state=final.transpose(1,2)
        else:
            offsets,bias,flags=(list(tail)+[None]*3)[:3]
            out=public.npu_causal_conv1d_fn(x.t().contiguous(),taps,bias,state,
                    query_start_loc=offsets,has_initial_state=flags,
                    activation=activation,null_block_id=None)
            out=out.t().contiguous()
        key="npu_causal_conv1d_fn"
        if public.BACKENDS.get(key)!="ctypes":
            raise RuntimeError("Wrong native causal backend")
        return out,state.transpose(1,2).contiguous()

    return Candidate(Implementation(run),"fla_npu.ops.ascendc.npu_causal_conv1d_fn"+("(single token with W-1 history)" if update else ""),
        "Transpose channel-major input/weights/history to native dim-last; preserve packed metadata/initial-state flags; private state per invocation. Decode uses FN on one token with the last W-1 history slots and returns the exact copy-only shifted W-slot history.",hashes)


def recurrent(case):
    import os
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    op = case._call.signature.name
    q = case.inputs[0]
    if q.dtype != torch.bfloat16:
        raise UnsupportedCase("Original RecurrentKda requires BF16 q/k/v; FP16 or FP32 to BF16 is lossy.")
    if max(q.shape[-1], case.inputs[2].shape[-1]) > 128:
        raise UnsupportedCase("Recurrent adapter currently maps feature dimensions <=128 by zero padding.")
    public, hashes = _native("aclnnRecurrentKda")
    params = case.params
    force_vfirst=os.environ.get('FLA_ALLPAIRS_NATIVE_STATE_VFIRST')=='1'

    def run(*args):
        if op == "DeltaNetRecurrentFwdOp":
            q, k, v, beta, initial = args
            q, k, v, beta = [t.unsqueeze(1) for t in (q, k, v, beta)]
            gate = torch.zeros((*q.shape[:2], v.shape[2], q.shape[-1]), device=q.device, dtype=torch.float32)
            offsets = offsets_cpu = a_log = dt_bias = None
        elif op == "DeltaNetFwdOp":
            q, k, v, beta, initial, offsets, offsets_cpu = args
            gate = torch.zeros((*q.shape[:2], v.shape[2], q.shape[-1]), device=q.device, dtype=torch.float32)
            a_log = dt_bias = None
        else:
            q, k, v, gate, beta, *tail = args
            initial, offsets, offsets_cpu, a_log, dt_bias = (tail + [None]*5)[:5]
            if op == "GDNFwdOp":
                gate = gate.unsqueeze(-1).expand(*gate.shape, q.shape[-1]).contiguous()
                if dt_bias is not None:
                    dt_bias = dt_bias.reshape(-1, 1).expand(-1, q.shape[-1]).contiguous()
        dk, dv = q.shape[-1], v.shape[-1]
        scale = 1.0 if op == "DeltaNetRecurrentFwdOp" else params.get("scale")
        scale = dk ** -0.5 if scale is None else scale
        q, k, v = F.pad(q, (0,128-dk)), F.pad(k, (0,128-dk)), F.pad(v,(0,128-dv))
        gate = F.pad(gate, (0,128-dk))
        if dt_bias is not None:
            dt_bias=F.pad(dt_bias.reshape(v.shape[2],dk),(0,128-dk)).contiguous()
        state_v_first = params.get("state_v_first", False)
        native_state_v_first=state_v_first or force_vfirst
        if initial is not None:
            initial = initial.float()
            initial = F.pad(initial, (0,128-dk,0,128-dv) if state_v_first else (0,128-dv,0,128-dk)).clone()
            if native_state_v_first!=state_v_first:initial=initial.transpose(-1,-2).contiguous()
        metadata = offsets_cpu if offsets_cpu is not None else offsets
        spans = [(slice(None),0,q.shape[1])] if metadata is None else [(slice(0,1),a,b) for a,b in zip(metadata.cpu().tolist(),metadata.cpu().tolist()[1:])]
        outputs, finals = [], []
        for seq,(batch,start,end) in enumerate(spans):
            state = None if initial is None else (initial if metadata is None else initial[seq:seq+1].contiguous())
            parts=[]
            # Original public recurrence supports up to eight tokens per call.
            # Split time only; every state transition executes inside RecurrentKda.
            for first in range(start,end,8):
                stop=min(first+8,end)
                tensors=[t[batch,first:stop].contiguous() for t in (q,k,v,gate,beta)]
                out,state=public.npu_recurrent_kda(*tensors,state,layout="BSND",scale=scale,
                    output_final_state=True,inplace_final_state=False,state_v_first=native_state_v_first,
                    use_qk_l2norm_in_kernel=params.get("use_qk_l2norm_in_kernel",False),
                    use_gate_in_kernel=params.get("use_gate_in_kernel",False),
                    use_beta_sigmoid_in_kernel=params.get("use_beta_sigmoid_in_kernel",False),
                    allow_neg_eigval=params.get("allow_neg_eigval",False),
                    safe_gate=params.get("lower_bound") is not None,lower_bound=params.get("lower_bound"),
                    A_log=a_log,dt_bias=dt_bias)
                parts.append(out)
            if state is None:
                state=torch.zeros((1,v.shape[2],128,128),device=q.device,dtype=torch.float32)
            outputs.append(torch.cat(parts,dim=1) if parts else v[batch,:0])
            finals.append(state)
        out=torch.cat(outputs,dim=0 if metadata is None else 1)[...,:dv].contiguous()
        state=torch.cat(finals,dim=0)
        if native_state_v_first!=state_v_first:state=state.transpose(-1,-2).contiguous()
        state=(state[...,:dv,:dk] if state_v_first else state[...,:dk,:dv]).contiguous()
        if public.BACKENDS.get("npu_recurrent_kda") != "ctypes":
            raise RuntimeError("Wrong native recurrent backend")
        return (out[:,0],state.to(args[0].dtype)) if op=="DeltaNetRecurrentFwdOp" else (out,state)

    return Candidate(Implementation(run),"fla_npu.ops.ascendc.npu_recurrent_kda",
        "Original KDA recurrence; Delta uses zero decay, GDN repeats scalar gate on channels. Feature pad/crop, private FP32 initial state, sequence/time slicing into <=8-token API calls; native beta/L2/raw-gate transforms preserved."+(' Explicit native value-first state layout with input/output transposes.' if force_vfirst else ''),hashes)


def gdn(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    p=case.params
    if p.get("use_gate_in_kernel",False):
        raise UnsupportedCase("ChunkGatedDeltaRuleFwd rejects raw gate; use separate recurrent deployment for BF16.")
    public, hashes=_native("aclnnChunkGatedDeltaRuleFwd")

    def run(q,k,v,g,beta,initial=None,offsets=None,offsets_cpu=None,a_log=None,dt_bias=None):
        dk,dv=q.shape[-1],v.shape[-1]
        scale=p.get("scale")
        scale=dk**-0.5 if scale is None else scale
        metadata=offsets_cpu if offsets_cpu is not None else offsets
        q,k,v=F.pad(q,(0,128-dk)),F.pad(k,(0,128-dk)),F.pad(v,(0,128-dv))
        state_v_first=p.get("state_v_first",False)
        state=None if initial is None else F.pad(initial,(0,128-dk,0,128-dv) if state_v_first else (0,128-dv,0,128-dk)).clone()
        bounds=None if metadata is None else metadata.cpu().tolist()
        if bounds is not None:
            longest=max(b-a for a,b in zip(bounds,bounds[1:]))
            def pack(t):
                return torch.stack([F.pad(t[0,a:b],(0,0)*(t.ndim-2)+(0,longest-(b-a))) for a,b in zip(bounds,bounds[1:])])
            q,k,v,g,beta=map(pack,(q,k,v,g,beta))
        layout="BSND"
        if q.dtype==torch.float16:
            q,k,v=[t.transpose(1,2).contiguous() for t in (q,k,v)]
            layout="BNSD"
        result=public.npu_chunk_gated_delta_rule_fwd(q,k,v,g,beta,initial_state=state,output_final_state=True,
            scale=scale,layout=layout,state_v_first=state_v_first,
            use_qk_l2norm_in_kernel=p.get("use_qk_l2norm_in_kernel",False),
            use_beta_sigmoid_in_kernel=p.get("use_beta_sigmoid_in_kernel",False),
            allow_neg_eigval=p.get("allow_neg_eigval",False))
        out=result[0]
        if bounds is not None:
            out=torch.cat([out[i,:b-a] for i,(a,b) in enumerate(zip(bounds,bounds[1:]))],dim=0).unsqueeze(0)
        final=result[1]
        if public.BACKENDS.get("npu_chunk_gated_delta_rule_fwd")!="ctypes":
            raise RuntimeError("Wrong native chunk backend")
        return out[...,:dv].contiguous(), (final[...,:dv,:dk] if state_v_first else final[...,:dk,:dv]).contiguous()

    return Candidate(Implementation(run),"fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd",
        "Original fused GDN, original gate/beta/L2 options and scale; feature pad/crop; FP16 head-major QKV; ragged sequences padded as independent dense batches with zero QKV/gate/beta.",hashes)
