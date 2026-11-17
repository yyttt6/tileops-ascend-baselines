"""Original FP32 TopP with complete original FLA local-cumsum APIs.

No device kernel definition, compiler flag, input dtype, checker or tolerance
changes. Full FP32 cumulative sums are assembled from two existing local-prefix
APIs; integer group positions use unique-index writes; no reduction or duplicate-index scatter.
"""
import torch
import torch.nn.functional as F
import pathlib,importlib.util,json,hashlib,sys
Q=pathlib.Path(__file__).resolve().parent

def factory(cumsum=None):
    hashes={}
    real=cumsum is None
    if real:
        meta=json.loads((Q/'TOPP_FLA_CDF_ORIGINAL_SOURCE.json').read_text())
        source=pathlib.Path(meta['source_file']).resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
        spec=importlib.util.spec_from_file_location('_fastest665_original_fla_cumsum',source)
        native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
        cumsum=native.chunk_local_cumsum
        hashes[str(source)]=meta['source_sha256']
    def top_p_mask(logits,p):
        batch,vocab=logits.shape
        probs=logits.float().softmax(-1)
        ordered,order=probs.sort(-1,descending=True)
        length=4096
        padded=F.pad(ordered,(0,(-vocab)%length))
        local=cumsum(padded.unsqueeze(-1),chunk_size=length,output_dtype=torch.float32).reshape(batch,-1,length)
        count=local.shape[1];extent=1<<(count-1).bit_length()
        totals=F.pad(local[:,:,-1].contiguous(),(0,extent-count)).unsqueeze(-1)
        prefix=cumsum(totals,chunk_size=extent,output_dtype=torch.float32)[...,0][:,:count]
        offsets=F.pad(prefix[:,:-1],(1,0))
        cumulative=(local+offsets[:,:,None]).reshape(batch,-1)[:,:vocab]
        exclusive=cumulative-ordered
        starts=torch.cat((torch.ones_like(ordered[:,:1],dtype=torch.bool),ordered[:,1:]!=ordered[:,:-1]),-1)
        positions=torch.arange(vocab,device=logits.device,dtype=torch.int32).expand(batch,vocab)
        group32=starts.to(torch.int32).cumsum(-1,dtype=torch.int32)-1
        groups=group32.long()
        # Exactly one start writes each meaningful group slot. Every other
        # position writes a distinct disjoint dummy slot: no duplicate indices.
        destinations=torch.where(starts,group32,positions+vocab).long()
        by_group=torch.empty((batch,2*vocab),dtype=torch.int32,device=logits.device)
        by_group.scatter_(-1,destinations,positions)
        first=by_group.gather(-1,groups).long()
        above_sorted=exclusive.gather(-1,first)
        # Kept probability groups form a prefix. Return the equivalent
        # probability threshold mask directly, retaining whole boundary ties.
        cutoff=ordered.masked_fill(above_sorted>=p.view(batch,1).to(logits.device),float('inf')).amin(-1)
        return logits.masked_fill(probs<cutoff[:,None],float('-inf'))
    if real:
        checker=pathlib.Path('/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs/benchmarks/ascend_all_pairs/jit_evidence.py')
        spec=importlib.util.spec_from_file_location('_fastest665_original_jit_evidence',checker)
        evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
        top_p_mask=evidence.triton_run(top_p_mask,hashes,Q/'topp-fla-jit-cache')
    return top_p_mask,hashes
