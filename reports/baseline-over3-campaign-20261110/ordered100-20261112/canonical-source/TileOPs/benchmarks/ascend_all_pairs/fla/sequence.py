"""Original FLA primitives for exact sequence-modeling subcontracts."""
from pathlib import Path


def chunk_cumsum(case):
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .gate_activations import gate_activation
    from .native_affine import make_affine
    from .providers import _triton_module
    p=case.params
    if not p.get('dt_softplus',False) or p.get('dt_min',0.0)!=0.0 or p.get('dt_max',float('inf'))!=float('inf'):
        raise UnsupportedCase('This native softplus route requires dt_softplus=True and bounds [0,+inf].')
    view=SimpleNamespace(params={},_call=SimpleNamespace(signature=SimpleNamespace(name='SoftplusFwdOp')))
    softplus=gate_activation(view)
    affine,hashes=make_affine()
    scan,source=_triton_module('cumsum')
    chunk=int(p['chunk_len'])
    dtype=getattr(torch,p['out_dtype'])

    def run(dt,A,dt_bias=None):
        value=dt if dt_bias is None else affine(dt,1.0,dt_bias)
        value=softplus.implementation.run(value)
        # Original softplus is nonnegative, so the specified [0,+inf] clamp
        # does not alter its mathematical output.
        batch,length,heads=value.shape
        packed=value.reshape(batch,length//chunk,chunk,heads).permute(0,3,1,2).contiguous()
        dA=affine(packed,A[None,:,None,None])
        prefix=scan.chunk_local_cumsum(dA.reshape(-1,chunk,1),chunk_size=chunk,output_dtype=torch.float32)
        return packed.to(dtype),prefix.reshape(packed.shape)

    return Candidate(Implementation(run),'FLA KdaGateCumsum(T=1 raw gate), native tap product, original chunk_local_cumsum',
        'Original native softplus and optional FP32 bias, then layout-only chunk packing. Keep FP32 dt for A multiplication and prefix accumulation while separately producing the manifest low-precision dt_out. Return both complete outputs, retaining original dt_softplus/bounds contract.',
        softplus.source_sha256|hashes|source_hashes(Path(__file__),source))


def mhc_post(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .native_affine import make_affine
    affine,hashes=make_affine()

    def run(x_layer_out,h_post,x_res):
        batch,width=x_layer_out.shape
        expanded=x_layer_out[:,None,:].float().expand(batch,h_post.shape[1],width)
        result=affine(expanded,h_post[:,:,None],x_res.reshape(batch,-1,width).float())
        return result.reshape(x_res.shape).to(x_res.dtype)

    return Candidate(Implementation(run),'FLA original FP32 causal tap/residual (MHC broadcast)',
        'Copy-only broadcast of layer output and per-expansion weights, then original native FP32 multiply/add with residual and one manifest output cast. All current values and copies handled inside run.',
        hashes|source_hashes(Path(__file__)))


def state_passing(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .exponential import exponential
    from .native_affine import make_affine
    exp=exponential(case)
    affine,hashes=make_affine()

    def run(states,dA_chunk_cumsum,initial_states=None):
        batch,chunks,heads,width=states.shape
        state=torch.zeros((batch,heads,width),device=states.device,dtype=torch.float32) if initial_states is None else initial_states.clone()
        scales=exp.implementation.run(dA_chunk_cumsum)
        previous=[]
        for chunk in range(chunks):
            previous.append(state)
            state=affine(state,scales[:,:,chunk,None],states[:,chunk].float())
        return torch.stack(previous,dim=1),state

    return Candidate(Implementation(run),'FLA ChunkFwdH exponentials + original FP32 causal tap/residual recurrence',
        'Match original SSD inter-chunk recurrence scale=exp(dA), state=scale*state+update. Return the state before every chunk and final FP32 state, including absent initial state. Original FLA kernels supply all exponentials/multiplication/addition; no saved input-derived data.',
        exp.source_sha256|hashes|source_hashes(Path(__file__)))


def chunk_coupling(case):
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .native_affine import make_affine
    from .spatial import gemm
    from .cube_matmul import make_matmul
    chunk=int(case.params['chunk_len'])
    x=case.inputs[0]
    affine,hashes=make_affine()
    if x.dtype==torch.bfloat16:
        matmul,more=make_matmul(torch.float32)
        hashes.update(more)
    else:
        view=SimpleNamespace(inputs=(x[0,:chunk,0,:],x[0,:chunk,0,:].t()),params={},_call=SimpleNamespace(signature=SimpleNamespace(name='GemmFwdOp')))
        native=gemm(view)
        matmul=native.implementation.run
        hashes.update(native.source_sha256)

    def run(C_mat,B_mat):
        batch,length,groups,width=C_mat.shape
        count=length//chunk
        left=C_mat.reshape(batch,count,chunk,groups,width).permute(0,1,3,2,4)
        right=B_mat.reshape(batch,count,chunk,groups,width).permute(0,1,3,4,2)
        # This is an input-independent causal layout constant, not value math.
        mask=torch.tensor([[1.0 if col<=row else 0.0 for col in range(chunk)] for row in range(chunk)],device=C_mat.device,dtype=torch.float32)
        results=[]
        for b in range(batch):
            pieces=[]
            for c in range(count):
                pieces.append(torch.stack([affine(matmul(left[b,c,g],right[b,c,g]).float(),mask).to(C_mat.dtype) for g in range(groups)]))
            results.append(torch.stack(pieces))
        return torch.stack(results)

    return Candidate(Implementation(run),'FLA native matrix product + original constant causal tap mask',
        'Chunk and group are batch dimensions; compute original native C @ B-transpose and apply input-independent lower-triangle zeros by original tap multiplication. BF16 uses native FP32 final state; FP16 uses original causal window-dot APIs. Packing, mask allocation and each original library call are inside run.',
        hashes|source_hashes(Path(__file__)))
