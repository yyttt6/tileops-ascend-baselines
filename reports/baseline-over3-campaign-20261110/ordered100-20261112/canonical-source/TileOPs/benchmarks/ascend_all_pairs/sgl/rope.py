"""Supply positional coefficient tables to SGL's original rotary kernel."""
from pathlib import Path


def rope_variant(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    from workloads import rope as tables
    name=case._call.signature.name
    p=case._call.params
    host='python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.fused_rope_qk_mqa','fused_rope_qk_mqa',host)
    table_fn={'RoPELlama31FwdOp':tables.llama31_frequency_tables,'YaRNFwdOp':tables.yarn_frequency_tables,'LongRoPEFwdOp':tables.longrope_frequency_tables,'RoPENeoxPositionIdsFwdOp':tables.rope_frequency_tables}[name]
    def run(x,position_ids=None):
        if name=='RoPENeoxPositionIdsFwdOp':
            dim=p.get('rotary_dim') or x.shape[-1]
            cos,sin=table_fn(dim,p['max_position'],base=p['base'],dtype=x.dtype,device=x.device)
            cache=torch.cat((cos,sin),dim=-1)[position_ids].contiguous().float()
            packed=x.contiguous().float()
        else:
            dim=x.shape[-1]
            seq=x.shape[0] if p['input_layout']=='1d' else x.shape[1]
            args={k:v for k,v in p.items() if k not in ('input_layout','rope_layout')}
            if 'rescale_factors' in case.workload.tensors:
                args['rescale_factors']=case.workload.tensors['rescale_factors']
            cos,sin=table_fn(dim,seq,dtype=x.dtype,device=x.device,**args)
            batch=1 if x.ndim==2 else x.shape[0]
            cache=torch.cat((cos,sin),dim=-1).repeat(batch,1).float()
            packed=x.float().contiguous().reshape(batch*seq,1 if x.ndim==2 else x.shape[2],dim)
        out,_=fn(packed,packed,cache,dim,p.get('rope_layout','neox')=='neox')
        return out.reshape_as(x).to(x.dtype)
    return Candidate(Implementation(run),'sgl_kernel_npu.norm.fused_rope_qk_mqa.fused_rope_qk_mqa',
                     'Generate declared positional coefficient metadata each run; gather explicit positions or repeat per batch; lossless FP32 input/table widening; all rotation arithmetic in original SGL kernel.',
                     source_hashes(Path(__file__),Path(tables.__file__),LIBRARIES/'sgl-kernel-npu'/host),
                     timing_mode='host_roundtrip' if name=='RoPELlama31FwdOp' else 'device_async',
                     limitations=('Coefficient generation shares original manifest table definitions; native rotation itself is independently executed. All table and layout costs included.',))
