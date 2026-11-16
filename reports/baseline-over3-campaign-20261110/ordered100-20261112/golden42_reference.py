"""Authorized CPU mathematical oracle; never a timed baseline implementation.

The caller must supply the frozen ordinal AND metadata. A certified manifest
binds the exact payload, descriptor, parameters, key and this source hash.
Historical reference_policy/overrides and prepared inputs remain immutable.
"""
import hashlib
import json
from pathlib import Path

import torch

SCOPE = frozenset([184,240,241,244,245,270,271,272,273,274,276,277,278,279,280,281,282,284,295,296,297,309,372,486,487,489,490,491,492,493,494,495,496,498,759,772,773,774,777,804,1082,1255])


def matrix(left, right, alternate=False):
    """Full K, row bounded FP64; alternative uses reverse split-K reduction."""
    assert left.device.type == right.device.type == 'cpu'
    assert left.ndim == right.ndim == 2 and left.shape[1] == right.shape[0]
    out = torch.empty((left.shape[0], right.shape[1]), dtype=left.dtype)
    right64 = right.double()
    for first in range(0, left.shape[0], 64):
        x = left[first:first+64].double()
        if alternate:
            acc = torch.zeros((x.shape[0], right.shape[1]), dtype=torch.float64)
            # Distinct K grouping and summation order; every K contributes once.
            for k in reversed(range(0, right.shape[0], 127)):
                acc += x[:, k:k+127] @ right64[k:k+127]
        else:
            acc = x @ right64
        out[first:first+64] = acc.to(left.dtype)
    return out


def mathematical_reference(op, params, args, alternate=False):
    assert all(not isinstance(x, torch.Tensor) or x.device.type == 'cpu' for x in args)
    if op == 'GemmFwdOp':
        a, b = args
        return matrix(a.T if params['trans_a'] else a,
                      b.T if params['trans_b'] else b, alternate)
    if op == 'BmmFwdOp':
        a, b = args
        return torch.stack([matrix(a[k], b[k], alternate) for k in range(a.shape[0])])
    if op == 'GroupedGemmFwdOp':
        a, b, sizes, offsets = args
        results = []
        first = 0
        for g, size in enumerate(sizes.tolist()):
            end = first + int(size)
            if params['transpose_a']:
                left = a[first:end].T
                right = b[:,first:end].T if params['transpose_b'] else b[first:end]
            else:
                left = a[first:end]
                right = b[g].T if params['transpose_b'] else b[g]
            results.append(matrix(left, right, alternate))
            first = end
        assert first == a.shape[0]
        return torch.stack(results) if params['transpose_a'] else torch.cat(results)
    if op == 'InstanceNormFwdOp':
        x, rm, rv, weight, bias = args
        assert rm is rv is None and params.get('use_input_stats', True)
        value = x.double().flatten(2)
        if alternate:
            # Welford parallel merge, independent of centered two-pass variance.
            count = 0
            mean = torch.zeros_like(value[..., :1])
            m2 = torch.zeros_like(mean)
            for block in reversed(value.split(37, dim=-1)):
                n = block.shape[-1]
                part_mean = block.mean(-1, keepdim=True)
                part_m2 = ((block-part_mean)**2).sum(-1, keepdim=True)
                delta = part_mean-mean
                total = count+n
                m2 = m2+part_m2+delta**2*(count*n/total)
                mean = mean+delta*(n/total)
                count = total
            var = m2/count
        else:
            mean = value.mean(-1, keepdim=True)
            var = ((value-mean)**2).mean(-1, keepdim=True)
        eps = params.get('eps', 1e-5)
        if eps is None:
            eps = torch.finfo(torch.float32).eps
        y = (value-mean)*torch.rsqrt(var+eps)
        if weight is not None:
            y = y*weight.double().reshape(1,-1,1)
        if bias is not None:
            y = y+bias.double().reshape(1,-1,1)
        return y.reshape(x.shape).to(x.dtype)
    if op == 'SSDChunkCouplingFwdOp':
        c, b = args
        n, _, g, state = c.shape
        q = params['chunk_len']
        assert c.shape[1] % q == 0
        nc = c.shape[1] // q
        cc = c.reshape(n,nc,q,g,state)
        bb = b.reshape(n,nc,q,g,state)
        if alternate:
            value = torch.empty((n,nc,g,q,q), dtype=c.dtype)
            for batch in range(n):
                for chunk in range(nc):
                    for group in range(g):
                        value[batch,chunk,group] = matrix(cc[batch,chunk,:,group],bb[batch,chunk,:,group].T, True)
        else:
            value = torch.einsum('bcqgn,bcsgn->bcgqs',cc.double(),bb.double()).to(c.dtype)
        mask = torch.tril(torch.ones(q,q,dtype=torch.bool))
        return value*mask
    raise ValueError('Outside authorized operator scope: '+op)


def corrected_reference_for_case(ordinal, meta, manifest=None):
    """Fail closed unless the exact case has a successful two-route certificate."""
    assert ordinal in SCOPE
    manifest = Path(manifest or Path(__file__).with_name('GOLDEN42_CERTIFICATES.json'))
    doc = json.loads(manifest.read_text())
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert doc['oracle_source_sha256'] == source_sha
    rec = next(r for r in doc['rows'] if r['ordinal'] == ordinal)
    assert rec['status'] == 'certified'
    for name in ('key','params','payload_sha256','descriptor_sha256'):
        assert rec[name] == meta[name], 'Golden identity mismatch: '+name
    def reference(*cpu_args):
        return mathematical_reference(meta['key'][1], meta['params'], cpu_args)
    return reference
