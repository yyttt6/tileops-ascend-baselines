"""Bind the original LayerNormAddAddQuant residual output as FP16 Add."""
import json
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes

_schema = None
_loaded = False


def transformer_add(case):
    import torch
    from benchmarks.api import Implementation

    if any(t.dtype != torch.float16 for t in case.inputs) or case.params.get('alpha', 1) != 1:
        raise UnsupportedCase('The original residual-add entry has FP16 pointers and unit input coefficients.')
    global _schema, _loaded
    root = LIBRARIES / 'ops-transformer'
    cache = WORKSPACE / '.cache/ascend-classic/native-pilot-20261020/ops-transformer-add'
    receipt_file = cache / 'build_receipt.json'
    receipt = json.loads(receipt_file.read_text())
    binary = cache / 'libclassic_transformer_add_original.so'
    actual = source_hashes(binary, *(WORKSPACE / n for n in receipt['original_source_sha256']))
    for name, digest in receipt['original_source_sha256'].items():
        if actual[name] != digest:
            raise RuntimeError('Original Add source changed since its isolated build: ' + name)
    if actual[str(binary.relative_to(WORKSPACE))] != receipt['binary_sha256']:
        raise RuntimeError('Original Add binary changed since its isolated build.')
    if not _loaded:
        if hasattr(torch.ops.ascend_ops, 'layernorm_add_add_quant'):
            raise RuntimeError('Another LayerNormAddAddQuant implementation is already registered.')
        # The source registers an implementation but omits its schema. Bind the
        # exact C++ signature; annotate the two output buffers that it mutates.
        _schema = torch.library.Library('ascend_ops', 'FRAGMENT')
        _schema.define(
            'layernorm_add_add_quant(int block_dim, Tensor inOne, Tensor inTwo, '
            'Tensor inThr, Tensor gamma, Tensor beta, Tensor scale, '
            'Tensor(a!) outLynQuant, Tensor(b!) outAdd, float epsilon, bool constrait) -> int')
        torch.ops.load_library(str(binary))
        owner = str(root / 'experimental/moe/layernorm_add_add_quant/layernorm_add_add_quant.cpp')
        table = torch._C._dispatch_dump_table('ascend_ops::layernorm_add_add_quant')
        if not any(line.startswith('PrivateUse1:') and owner in line for line in table.splitlines()):
            raise RuntimeError('Add binding did not resolve to the original native source.')
        _loaded = True
    fn = torch.ops.ascend_ops.layernorm_add_add_quant

    def run(x, y):
        if x.dtype != torch.float16 or y.dtype != torch.float16:
            raise UnsupportedCase('Native residual pointers are FP16.')
        x, y = torch.broadcast_tensors(x, y)
        shape, count = x.shape, x.numel()
        # Elementwise residual output is independent of the normalization rows.
        # Regroup elements to keep the original fixed UB allocation small.
        width = 1024
        padded = (count + width - 1) // width * width
        a, b = x.contiguous().reshape(-1), y.contiguous().reshape(-1)
        if padded != count:
            a = torch.nn.functional.pad(a, (0, padded - count))
            b = torch.nn.functional.pad(b, (0, padded - count))
        a, b = a.reshape(-1, width), b.reshape(-1, width)
        # x + (-0) preserves both signs of zero before the original second Add.
        neutral = torch.full((width,), -0.0, dtype=x.dtype, device=x.device)
        gamma = torch.ones_like(neutral)
        beta = torch.zeros_like(neutral)
        scale = torch.ones((width,), dtype=torch.float32, device=x.device)
        quant = torch.empty(a.shape, dtype=torch.int8, device=x.device)
        out = torch.empty_like(a)
        status = fn(min(40, a.shape[0]), a, neutral, b, gamma, beta, scale,
                    quant, out, 1e-5, False)
        if status != 0:
            raise RuntimeError('Original LayerNormAddAddQuant rejected its UB requirement.')
        return out.reshape(-1)[:count].reshape(shape)

    return Candidate(
        Implementation(run), 'ops-transformer::layernorm_add_add_quant.residual_output',
        'Original complete fused entry and raw residual output; FP16 only. Broadcast/reshape into width1024 rows, neutral negative-zero bias, crop/reshape output. Compiler target/header setup only; original math, auto-sync, buffering and kernel source unchanged.',
        source_hashes(Path(__file__), receipt_file) | actual,
        limitations=('Full CPU Add reference and existing nonfinite/signed-zero/input-preservation checks required; complete fused operation and packing cost belong to this candidate.',),
    )
