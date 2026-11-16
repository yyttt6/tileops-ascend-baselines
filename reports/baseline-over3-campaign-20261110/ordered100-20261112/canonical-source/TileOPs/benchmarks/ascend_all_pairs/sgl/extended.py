"""Adapters for the shared factory-completion inventory revision."""
from pathlib import Path


def causal_conv(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,local_function,source_hashes,LIBRARIES,UnsupportedCase
    host='python/sgl_kernel_npu/sgl_kernel_npu/mamba/causal_conv1d.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.mamba.causal_conv1d','causal_conv1d_update_v2',host)
    if not 2<=case.inputs[1].shape[-1]<=6:
        raise UnsupportedCase('Original SGL Triton update kernel has explicit weight taps for width2..6 only.')
    update=case._call.signature.name=='CausalConv1dUpdateFwdOp'
    def decode(x,weight,conv_state,bias=None):
        history=conv_state[...,1:].transpose(1,2).contiguous()
        indices=torch.arange(x.shape[0],device=x.device,dtype=torch.int32)
        y=fn(x.clone().contiguous(),history,weight.t().contiguous(),bias,case.params['activation'],conv_state_indices=indices)
        final=torch.cat((conv_state[...,1:],x.unsqueeze(-1)),dim=-1)
        return y,final
    def prefill(x,weight,conv_state,cu_seqlens,bias=None,has_initial_state=None):
        bounds=cu_seqlens.cpu().tolist()
        flags=[False]*(len(bounds)-1) if has_initial_state is None else has_initial_state.cpu().tolist()
        ys=[];states=[]
        for i,(begin,end) in enumerate(zip(bounds[:-1],bounds[1:])):
            history=conv_state[i:i+1].transpose(1,2).contiguous().clone()
            if not flags[i]:history.zero_()
            if end==begin:
                ys.append(x[:,begin:end]);states.append(history.transpose(1,2));continue
            xx=x[:,begin:end].t().unsqueeze(0).contiguous().clone()
            index=torch.zeros((1,),device=x.device,dtype=torch.int32)
            y=fn(xx,history,weight.t().contiguous(),bias,case.params['activation'],conv_state_indices=index)
            ys.append(y[0].t());states.append(history.transpose(1,2))
        return torch.cat(ys,dim=-1).contiguous(),torch.cat(states,dim=0).contiguous()
    return Candidate(Implementation(decode if update else prefill),'sgl_kernel_npu.mamba.causal_conv1d.causal_conv1d_update_v2',
        'Original Triton multi-token update API, independent dense sequence calls for packed prefill; transpose channels/taps/history, clone mutated inputs, metadata-controlled zero initial history. Return complete convolution and exact updated history. Native F.conv1d reference/fallback is never used.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),timing_mode='host_roundtrip' if not update else 'device_async')


def qk_norm_rope(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,local_function,source_hashes,LIBRARIES
    host='python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.split_qkv_rmsnorm_rope','split_qkv_rmsnorm_rope',host)
    p=case.params;dim=case.inputs[1].numel()
    def run(qkv,q_weight,k_weight,cache,positions):
        cos,sin=cache[positions.long()].chunk(2,dim=-1)
        cos=torch.cat((cos,cos),dim=-1).contiguous()
        sin=torch.cat((sin,sin),dim=-1).contiguous()
        q,k,v=fn(qkv.contiguous(),sin,cos,p['num_heads']*dim,p['num_kv_heads']*dim,dim,p['eps'],q_weight.contiguous(),k_weight.contiguous(),is_neox_style=p['rope_layout']=='neox')
        return torch.cat((q,k,v),dim=-1)
    return Candidate(Implementation(run),'sgl_kernel_npu.norm.split_qkv_rmsnorm_rope.split_qkv_rmsnorm_rope',
        'Gather actual position tables and duplicate half-width cos/sin for original API layout; preserve native FP32 normalization through rotation and concatenate all original Q/K/V outputs, including exact unchanged V.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
