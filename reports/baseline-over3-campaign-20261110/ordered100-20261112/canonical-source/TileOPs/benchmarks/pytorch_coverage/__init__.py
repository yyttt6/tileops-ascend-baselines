"""Ordinary PyTorch baseline bindings, separate from independent CPU oracles.

The candidate operates on the original tensor devices. No vendor baseline is
selected as a fallback. Registration does not imply measured correctness.
"""
from importlib import import_module

GROUPS = {
    'gdn': ('GDNFwdOp',),
    'kda': ('KDAFwdOp',),
    'linear': ('DeltaNetFwdOp', 'GLAFwdOp', 'DeltaNetChunkFwdOp'),
    'extended': ('CausalConv1dFwdOp', 'CausalConv1dUpdateFwdOp',
                 'GemmINT8W8A8FwdOp', 'MergeAttentionStatesFwdOp'),
    'attention': ('MLAVarlenFwdOp', 'GQAPrefillPagedWithKVCacheFwdOp', 'PagedKVCacheWriteFwdOp'),
    'moe_ssd': ('FusedMoESharedExpertFwdOp', 'SSDChunkStateFwdOp'),
    'fft_pool': ('FFTC2CFwdOp', 'AdaptiveMaxPool2dFwdOp', 'AdaptiveMaxPool2dIndicesFwdOp', 'CumprodFwdOp'),
}
OP_GROUP = {op: group for group, ops in GROUPS.items() for op in ops}
PREVIOUS_OPS = frozenset((
    'DeltaNetChunkBwdOp', 'GLAChunkBwdOp', 'GemmW4A8FwdOp', 'DSAPagedFwdOp',
    'FusedAddRMSNormPerTokenQuantFwdOp', 'FusedQKNormRoPEFwdOp',
    'HadamardTransformFwdOp', 'MLAKVCacheWriteFwdOp', 'MLAPagedFwdOp',
    'PagedKVCacheGatherFwdOp', 'SelectiveScanFwdOp',
))


def registered():
    return frozenset(OP_GROUP) | PREVIOUS_OPS


def module_for(op):
    if op in OP_GROUP:
        return import_module(f'benchmarks.pytorch_coverage.{OP_GROUP[op]}')
    if op in PREVIOUS_OPS:
        return import_module('benchmarks.pytorch_missing')
    raise KeyError(f'No explicit PyTorch candidate registered for {op}')


def bind(case):
    op = case._call.signature.name
    module = module_for(op)
    return module.bind(op, case._call.params) if op in PREVIOUS_OPS else module.bind(case)


def mutated_positions(case):
    op = case._call.signature.name
    if op in PREVIOUS_OPS:
        return {'FusedQKNormRoPEFwdOp': (0,), 'MLAKVCacheWriteFwdOp': (2,),
                'PagedKVCacheGatherFwdOp': (0,)}.get(op, ())
    return tuple(getattr(module_for(op), 'MUTATED', {}).get(op, ()))


def implementation(case):
    """Standard benchmark binding with reset for declared in-place inputs."""
    from benchmarks.api import Implementation
    from benchmarks.baselines import private_inputs
    run = bind(case)
    positions = mutated_positions(case)
    return private_inputs(run, case.inputs, *positions) if positions else Implementation(run=run)


def make_inputs(case, device):
    op = case._call.signature.name
    module = module_for(op)
    factory = getattr(module, 'make_inputs', None) if op not in PREVIOUS_OPS else None
    return tuple(factory(case, device)) if factory is not None else tuple(case.inputs)


def cpu_reference(case, inputs):
    """Oracle dispatch for an isolated CPU checker, never called by a candidate."""
    op = case._call.signature.name
    if op == 'GemmW4A8FwdOp':
        from benchmarks.pytorch_missing.validation import w4a8_reference
        return w4a8_reference(*inputs, group_size=case._call.params['group_size'])
    module = module_for(op)
    reference = getattr(module, 'cpu_reference', None) if op not in PREVIOUS_OPS else None
    if reference is not None:
        return reference(case, inputs)
    if case.reference is None:
        raise ValueError(f'{op} has no independent CPU reference')
    return case.reference(*inputs)
