"""Complete original top-k API followed by the validated full TopP route.

No device kernels, compiler flags, input precision, or validators are changed.
The TopP helper retains FP32 arithmetic and the original tie-group policy.
"""
from workloads.sampling import top_k_mask
from top_p_fla_unique_scatter_v1 import factory as top_p_factory


def factory(cumsum=None):
    top_p, hashes = top_p_factory(cumsum=cumsum)

    def top_k_top_p_mask(logits, k, p):
        return top_p(top_k_mask(logits, k), p)

    return top_k_top_p_mask, hashes
