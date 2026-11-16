"""Boolean OR through the existing original native maximum output."""
from .adapters import _reduce_rows
from .native import load_native


def logical_or(case):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    from benchmarks.api import Implementation
    p = case.params
    if p['input'][1] != 'bool' or p['other'][1] != 'bool':
        raise UnsupportedCase('Only boolean0/1 widens losslessly to the native BF16 maximum domain; arbitrary truth predicates are not native.')
    native, hashes = load_native()
    shape = tuple(p['output'][0])

    def run(x, other):
        import torch
        for tensor, name in ((x, 'input'), (other, 'other')):
            if tuple(tensor.shape) != tuple(p[name][0]) or tensor.dtype != torch.bool:
                raise ValueError('Original boolean OR input contract changed: ' + name)
        left = x.expand(shape).contiguous().reshape(-1).to(torch.bfloat16)
        right = other.expand(shape).contiguous().reshape(-1).to(torch.bfloat16)
        result = _reduce_rows(native, torch.stack((left, right), dim=1), 'max')
        return result.to(torch.bool).reshape(shape)

    return Candidate(Implementation(run), 'FlashMLA::sparse_prefill_fwd(max_logits)',
                     'Broadcast and pack complete bool operands using exact0/1 BF16 representation; original max statistic; exact0/1 cast back to bool. All input-dependent layout/copies occur inside run.', hashes,
                     limitations=('Original fixed H64/D512 attention also executes when selecting the max statistic. No performance measurement is claimed.',))
