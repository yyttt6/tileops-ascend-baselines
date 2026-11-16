"""Complete tracked/eval normalization using unchanged ops-math graph APIs."""
from math import prod
from pathlib import Path
from types import SimpleNamespace
from .specs import files


def make(case, op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes, UnsupportedCase
    from .graph import make as graph_make, deployment_hashes

    if op not in ('InstanceNormFwdOp', 'BatchNormFwdOp'):
        raise UnsupportedCase(op)
    p = case.params
    children = {}

    def call(name, *args, dim=None):
        shape = torch.broadcast_shapes(*(x.shape for x in args))
        params = {}
        if dim is not None:
            dim = tuple(d % args[0].ndim for d in dim)
            shape = tuple(1 if i in dim else n for i, n in enumerate(args[0].shape))
            params.update(dim=dim, keepdim=True)
        params['output'] = (tuple(shape), 'float32')
        if name == 'VarMean':
            params['correction'] = 0
        key = (name, tuple(tuple(x.shape) for x in args), dim)
        if key not in children:
            children[key] = graph_make(SimpleNamespace(params=params), name + 'FwdOp')
        return children[key].implementation.run(*args)

    def constant(x, value):
        return torch.full((), value, dtype=torch.float32, device=x.device)

    def add(x, y):
        return call('Sub', x, call('Neg', y))

    def scale(x, factor):
        return call('Mul', x, constant(x, factor))

    def run(original, running_mean, running_var, weight, bias):
        x = original.float()
        n, c = x.shape[:2]
        instance = op == 'InstanceNormFwdOp'
        use_input = p['use_input_stats'] if instance else p['training']
        channel_shape = (1, c) + (1,) * (x.ndim - 2)
        axes = tuple(range(2, x.ndim)) if instance else (0, *range(2, x.ndim))
        count = prod(x.shape[d] for d in axes)
        eps = p.get('eps')
        eps = torch.finfo(torch.float32).eps if eps is None else eps
        result_mean, result_var = running_mean, running_var
        if use_input:
            variance, mean = call('VarMean', x, dim=axes)
            if running_mean is not None:
                if count <= 1:
                    raise UnsupportedCase('Tracked variance requires at least two samples per reduction.')
                batch_mean = call('Mean', mean, dim=(0,)).reshape(c) if instance else mean.reshape(c)
                unbiased = scale(variance, count / (count - 1))
                batch_var = call('Mean', unbiased, dim=(0,)).reshape(c) if instance else unbiased.reshape(c)
                momentum = p['momentum']
                result_mean = add(scale(running_mean.float(), 1 - momentum), scale(batch_mean, momentum)).to(running_mean.dtype)
                result_var = add(scale(running_var.float(), 1 - momentum), scale(batch_var, momentum)).to(running_var.dtype)
        else:
            if running_mean is None or running_var is None:
                raise UnsupportedCase('Evaluation requires both stored running statistics.')
            mean = running_mean.float().reshape(channel_shape)
            variance = running_var.float().reshape(channel_shape)
        result = call('Mul', call('Sub', x, mean), call('Rsqrt', add(variance, constant(x, eps))))
        if weight is not None:
            result = call('Mul', result, weight.float().reshape(channel_shape))
        if bias is not None:
            result = add(result, bias.float().reshape(channel_shape))
        result = result.to(original.dtype)
        # Return statistics as outputs; never mutate the caller's tensors.
        return result if instance and running_mean is None else (result, result_mean, result_var)

    def cleanup():
        for child in children.values():
            child.cleanup()

    primitives = ('VarMean', 'Mean', 'Mul', 'Sub', 'Neg', 'Rsqrt')
    sources = [f for name in primitives for f in files(name + 'FwdOp')]
    return Candidate(Implementation(run), 'ops-math::native normalization with complete running statistics',
                     'Original native reductions and arithmetic; unbiased running variance and declared momentum; stored statistics in eval; all output tensors checked and original inputs preserved.',
                     source_hashes(Path(__file__), Path(__file__).with_name('graph.py'), *sources) | deployment_hashes(),
                     cleanup=cleanup, limitations=('Private staged route until complete device and native attribution checks.',))
