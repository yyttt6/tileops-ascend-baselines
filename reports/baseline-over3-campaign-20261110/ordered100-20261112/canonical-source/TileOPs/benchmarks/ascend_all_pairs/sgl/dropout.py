"""Original categorical kernel supplies Bernoulli decisions for native dropout."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .sampling import categorical
    from .additional import mul
    from .extra_arithmetic import _affine
    sampler = categorical(case)
    product = mul(case)
    affine, hashes = _affine(case)
    p = case.params['p'] if case.params.get('training',True) else 0.0
    seed_value = case.params.get('seed',0)
    def run(x):
        if p in (0.0,1.0):
            return affine(x,torch.zeros_like(x),1.0-p)
        flat = x.contiguous().reshape(-1)
        output = torch.empty_like(flat)
        seed = torch.tensor(seed_value,device=x.device,dtype=torch.int64)
        probabilities = torch.tensor([p,1.0-p],device=x.device,dtype=torch.float32)
        for start in range(0,flat.numel(),65536):
            values = flat[start:start+65536].float()
            probs = probabilities.expand(values.numel(),2).contiguous()
            offset = torch.tensor(start,device=x.device,dtype=torch.int64)
            keep = sampler.implementation.run(probs,seed,offset).float()
            masked = product.implementation.run(values,keep)
            scaled = affine(masked,torch.zeros_like(masked),1.0/(1.0-p))
            output[start:start+values.numel()].copy_(scaled.to(x.dtype))
        return output.reshape(x.shape)
    return Candidate(Implementation(run),sampler.api+' + original scale_shift/mul_add dropout scaling',
        'The complete original categorical sampler selects Bernoulli drop/keep from [p,1-p] for every element. Original native product and reciprocal-keep-probability affine scaling compute output. Independent element batches bound memory; seed/offset uniform preparation is inherited unchanged from the native sampler adapter. The original dropout verifier explicitly accepts independent random generators by scale and six-sigma mask-rate checks.',
        sampler.source_sha256 | product.source_sha256 | hashes | source_hashes(Path(__file__)),
        timing_mode='host_roundtrip',
        limitations=('Retains the original statistical contract; no claim that a separate generator reproduces the reference mask bits.',))
