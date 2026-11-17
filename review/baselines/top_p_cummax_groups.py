"""Equivalent full Torch nucleus filter using a native prefix-maximum API.

Original FP32 softmax, ordering, cumsum-minus-value, group-first selection,
scatter and final typed mask are retained. Only SearchSorted group indexing is
replaced by exact integer prefix maxima of group starts. No kernel edits.
"""
import torch

def top_p_mask(logits,p):
    batch,vocab=logits.shape
    probs=logits.float().softmax(-1)
    ordered,order=probs.sort(-1,descending=True)
    exclusive=ordered.cumsum(-1)-ordered
    starts=torch.cat((torch.ones_like(ordered[:,:1],dtype=torch.bool),
                      ordered[:,1:]!=ordered[:,:-1]),dim=-1)
    positions=torch.arange(vocab,device=logits.device,dtype=torch.int32).expand(batch,vocab)
    first=torch.where(starts,positions,torch.zeros_like(positions)).cummax(-1).values.long()
    above=torch.empty_like(probs).scatter_(-1,order,exclusive.gather(-1,first))
    return logits.masked_fill(above>=p.view(batch,1).to(logits.device),float('-inf'))
