"""BF16 matrix products from the original ChunkFwdH final FP32 state."""
from pathlib import Path


def make_matmul(output_dtype=None):
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import source_hashes
    from .providers import _native
    native,hashes=_native('aclnnChunkFwdH')
    output_dtype=torch.bfloat16 if output_dtype is None else output_dtype

    def matmul(left,right):
        if left.dtype!=torch.bfloat16 or right.dtype!=torch.bfloat16:
            raise TypeError('Original ChunkFwdH requires BF16 operands; no lossy conversion allowed.')
        rows,width=left.shape;columns=right.shape[1]
        result=torch.empty((rows,columns),device=left.device,dtype=output_dtype)
        tiles=[(i,min(rows,i+128),j,min(columns,j+128)) for i in range(0,rows,128) for j in range(0,columns,128)]
        # Limit duplicated K/V packing plus saved native chunk states. Tile
        # grouping is API batch packing; original native schedules stay intact.
        group_size=max(1,min(16,(64*1024*1024)//max(1,width*128*2)))
        for first in range(0,len(tiles),group_size):
            group=tiles[first:first+group_size]
            key=torch.stack([F.pad(left[i:stop].t().contiguous(),(0,128-(stop-i))) for i,stop,j,end in group]).unsqueeze(1)
            value=torch.stack([F.pad(right[:,j:end].contiguous(),(0,128-(end-j))) for i,stop,j,end in group]).unsqueeze(1)
            gate=torch.zeros((len(group),1,width),device=left.device,dtype=torch.float32)
            _,_,state=native.npu_chunk_fwd_h(key,torch.zeros_like(key),value,g=gate,
                initial_state=None,output_final_state=True,use_exp2=False)
            for index,(i,stop,j,end) in enumerate(group):
                result[i:stop,j:end]=state[index,0,:stop-i,:end-j].to(output_dtype)
        if native.BACKENDS.get('npu_chunk_fwd_h')!='ctypes':raise RuntimeError('Wrong original ChunkFwdH backend')
        return result

    return matmul,hashes|source_hashes(Path(__file__))


def gemm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    matmul,hashes=make_matmul()
    p=case.params;batched=case._call.signature.name=='BmmFwdOp'
    def run(a,b):
        a=a.transpose(-1,-2) if p.get('trans_a',False) else a
        b=b.transpose(-1,-2) if p.get('trans_b',False) else b
        return torch.stack([matmul(aa,bb) for aa,bb in zip(a,b,strict=True)]) if batched else matmul(a,b)
    return Candidate(Implementation(run),'FLA npu_chunk_fwd_h(W=0,g=0, final FP32 state)',
        'Map reduction K to native sequence length, each 128x128 output tile to an independent batch. Original state recurrence sums A-transpose/B outer products in its FP32 final state; cast only the complete result to manifest BF16. Copy-only transpose/pad/batch packing, bounded duplicated operands, no split-K output rounding or native schedule edits.',hashes)
