"""Compositions of unchanged ops-math graph APIs; Torch only supplies storage/layout."""
from math import prod, sqrt
from pathlib import Path
from types import SimpleNamespace
from .specs import files

from .composition_specs import COMPOSITIONS
PRIMITIVES = ('Neg', 'Sub', 'Mul', 'Div', 'Exp', 'Expm1', 'Log1p', 'Log',
              'Reciprocal', 'Erf', 'Tanh', 'Rsqrt', 'Maximum', 'Minimum',
              'Gt', 'Where', 'Amax', 'Sum', 'Mean')


def make(case, op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .graph import make as graph_make, deployment_hashes

    p = case.params
    children = {}
    norm = None
    if op in ('FusedAddLayerNormFwdOp', 'AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp'):
        from .layernorm import make as norm_make
        norm = norm_make(SimpleNamespace(params=p), 'LayerNormFwdOp')

    def call(name, *args, dim=None):
        shape = torch.broadcast_shapes(*(x.shape for x in args))
        kwargs = {}
        if dim is not None:
            axes = tuple(d % args[0].ndim for d in (dim if isinstance(dim, tuple) else (dim,)))
            shape = tuple(1 if i in axes else n for i, n in enumerate(args[0].shape))
            kwargs.update(dim=axes, keepdim=True)
        dtype = 'bool' if name == 'Gt' else 'float32'
        key = (name, tuple(tuple(x.shape) for x in args), tuple(shape), str(dim))
        if key not in children:
            children[key] = graph_make(SimpleNamespace(params={
                'output': (tuple(shape), dtype), **kwargs}), name + 'FwdOp')
        return children[key].implementation.run(*args)

    def scalar(x, value):
        return torch.full((), value, dtype=torch.float32, device=x.device)

    def add(x, y):
        return call('Sub', x, call('Neg', y))

    def mul(x, value):
        return call('Mul', x, value if isinstance(value, torch.Tensor) else scalar(x, value))

    def sigmoid(x):
        return call('Reciprocal', add(call('Exp', call('Neg', x)), scalar(x, 1)))

    def gelu(x, tanh=False):
        if tanh:
            cubic = mul(mul(x, x), x)
            inner = mul(add(x, mul(cubic, 0.044715)), sqrt(2 / 3.141592653589793))
            factor = call('Tanh', inner)
        else:
            factor = call('Erf', mul(x, 1 / sqrt(2)))
        return mul(mul(x, 0.5), add(factor, scalar(x, 1)))

    def softplus(x, beta=1., threshold=20.):
        scaled = mul(x, beta)
        curved = mul(call('Log1p', call('Exp', scaled)), 1 / beta)
        return call('Where', call('Gt', scaled, scalar(x, threshold)), x, curved)

    def rms(x, weight):
        dims = tuple(range(x.ndim - len(p.get('normalized_shape', (x.shape[-1],))), x.ndim))
        eps = p.get('eps')
        eps = torch.finfo(torch.float32).eps if eps is None else eps
        inv = call('Rsqrt', add(call('Mean', mul(x, x), dim=dims), scalar(x, eps)))
        result = mul(x, inv)
        return result if weight is None else mul(result, weight.float())

    def run(*args):
        original = args[0]
        x = original.float()
        if op in ('SiluAndMulFwdOp', 'GeluAndMulFwdOp', 'GeluTanhAndMulFwdOp'):
            width = x.shape[-1] // 2
            gate, value = x[..., :width], x[..., width:]
            activated = mul(gate, sigmoid(gate)) if op == 'SiluAndMulFwdOp' else gelu(gate, op == 'GeluTanhAndMulFwdOp')
            result = mul(activated, value)
        elif op == 'SigmoidFwdOp':
            result = sigmoid(x)
        elif op == 'SiluFwdOp':
            result = mul(x, sigmoid(x))
        elif op == 'GeluFwdOp':
            result = gelu(x, p.get('approximate', 'none') == 'tanh')
        elif op in ('HardsigmoidFwdOp', 'HardswishFwdOp'):
            ramp = call('Minimum', call('Maximum', add(x, scalar(x, 3)), scalar(x, 0)), scalar(x, 6))
            result = mul(ramp, 1 / 6)
            if op == 'HardswishFwdOp':
                result = mul(x, result)
        elif op in ('EluFwdOp', 'SeluFwdOp'):
            alpha = 1.6732632423543772 if op == 'SeluFwdOp' else p.get('alpha', 1.)
            negative = mul(call('Expm1', x), alpha)
            result = call('Where', call('Gt', x, scalar(x, 0)), x, negative)
            if op == 'SeluFwdOp':
                result = mul(result, 1.0507009873554805)
        elif op in ('LeakyReluFwdOp', 'PreluFwdOp'):
            if op == 'PreluFwdOp':
                weight = args[1].float()
                if weight.numel() > 1:
                    weight = weight.reshape((1, weight.numel()) + (1,) * (x.ndim - 2))
            else:
                weight = scalar(x, p.get('negative_slope', .01))
            result = call('Where', call('Gt', x, scalar(x, 0)), x, mul(x, weight))
        elif op == 'SoftplusFwdOp':
            result = softplus(x, p.get('beta', 1.), p.get('threshold', 20.))
        elif op == 'MishFwdOp':
            result = mul(x, call('Tanh', softplus(x)))
        elif op in ('SoftmaxFwdOp', 'LogSoftmaxFwdOp'):
            dim = p.get('dim', -1)
            centered = call('Sub', x, call('Amax', x, dim=dim))
            exponential = call('Exp', centered)
            denominator = call('Sum', exponential, dim=dim)
            result = call('Div', exponential, denominator) if op == 'SoftmaxFwdOp' else call('Sub', centered, call('Log', denominator))
        elif op == 'RMSNormFwdOp':
            result = rms(x, args[1])
        elif op in ('FusedAddRMSNormFwdOp', 'FusedAddLayerNormFwdOp'):
            residual = add(x, args[1].float()).to(original.dtype)
            result = rms(residual.float(), args[2]) if op == 'FusedAddRMSNormFwdOp' else norm.implementation.run(residual, *args[2:])
            return result.to(original.dtype), residual
        elif op in ('AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp'):
            # The original row kernel takes one shared affine vector. Neutral
            # affine permits a single call for all rows; original graph APIs
            # then apply the manifest's arbitrary per-element scale and shift.
            result = add(mul(norm.implementation.run(x, None, None), args[1].float()), args[2].float())
            if op == 'AdaLayerNormZeroFwdOp':
                result = mul(result, args[3].float())
        else:
            raise UnsupportedCase(op)
        return result.to(getattr(torch, p['output'][1]))

    def cleanup():
        for child in children.values():
            child.cleanup()
        if norm is not None and norm.cleanup is not None:
            norm.cleanup()

    sources = [f for name in PRIMITIVES for f in files(name + 'FwdOp')]
    hashes = source_hashes(Path(__file__), Path(__file__).with_name('graph.py'), *sources) | deployment_hashes()
    if norm is not None:
        hashes.update(norm.source_sha256)
    return Candidate(Implementation(run), 'ops-math::original native API composition for ' + op,
                     'Every arithmetic, comparison and selection uses unchanged own-library graph APIs. Torch supplies exact FP32 widening, views, constants and final output casts. No input-dependent data cached.',
                     hashes, cleanup=cleanup,
                     limitations=('Require full manifest outputs, input preservation and attribution for every native primitive family used.',))
