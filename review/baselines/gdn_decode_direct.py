"""Full FP32 one-token GDN recurrence using existing Torch APIs, functional state."""
import torch

def factory(params,device='npu'):
    qs=params['q'][0];vs=params['v'][0]
    assert qs[1]==vs[1]==1 and vs[2]%qs[2]==0
    assert params['q'][1]==params['k'][1]==params['v'][1]=='float16'
    assert params['initial_state'][1]=='float32'
    for flag in ['use_qk_l2norm_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval','state_v_first','use_gate_in_kernel']:
        assert not params[flag]
    group=vs[2]//qs[2]
    scale=params['scale'] if params['scale'] is not None else qs[-1]**-0.5
    scale_tensor=torch.tensor([scale],device=device,dtype=torch.float32)
    def run(q,k,v,g,beta,initial_state,cu_seqlens=None,cu_seqlens_cpu=None,a_log=None,dt_bias=None):
        assert list(q.shape)==qs and list(v.shape)==vs
        assert all(x is None for x in [cu_seqlens,cu_seqlens_cpu,a_log,dt_bias])
        assert initial_state.dtype==torch.float32
        qt=q[:,0].float().repeat_interleave(group,dim=1)*scale_tensor
        kt=k[:,0].float().repeat_interleave(group,dim=1)
        decay=g[:,0].float().exp()
        old_value=torch.einsum('bhkv,bhk->bhv',initial_state,kt)
        written=beta[:,0].float().unsqueeze(-1)*(v[:,0].float()-decay.unsqueeze(-1)*old_value)
        # Every operation is out of place: preserve the real input state without
        # the generic reference's identity indexing and final identity scatter.
        final=decay[...,None,None]*initial_state+kt.unsqueeze(-1)*written.unsqueeze(-2)
        output=torch.einsum('bhk,bhkv->bhv',qt,final).unsqueeze(1).to(q.dtype)
        return output,final
    return run
