"""DeepEP's own kernels; importing this registry never imports a device runtime."""

from .adapters import add, count_nonzero, paged_cache_gather, permute_align, prepermute

PROVIDERS = {
    "AddFwdOp": add,
    "CountNonzeroFwdOp": count_nonzero,
    "MoEPrePermuteFwdOp": prepermute,
    "MoEPermuteAlignFwdOp": permute_align,
    "PagedKVCacheGatherFwdOp": paged_cache_gather,
}
