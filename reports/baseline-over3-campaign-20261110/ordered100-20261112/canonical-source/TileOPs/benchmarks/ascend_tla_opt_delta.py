"""Focused original opt_gdn pipeline, separate from the quarantined gdn entry."""
import ast
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_tla_baselines import _runtime
from benchmarks.ascend_tla_final import _load


def tla_opt_deltanet(case):
    import torch
    from benchmarks.api import Implementation

    q, k, v, beta, *extra = case.inputs
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if (q.dtype not in (torch.float16, torch.bfloat16)
            or any(x.dtype != q.dtype for x in (k, v, beta))
            or any(x is not None for x in extra) or case.params.get('use_qk_l2norm_in_kernel')
            or tuple(k.shape) != tuple(q.shape) or tuple(v.shape[:3]) != tuple(q.shape[:3])
            or tuple(beta.shape) != tuple(q.shape[:3]) or heads % 2 or length % 256
            or (key_dim, value_dim) != (64, 64)):
        raise UnsupportedCase('Focused original opt_gdn entry requires dense zero-state equal-head64 contracts and complete256-token groups.')
    root = 'examples/linear_attention_and_rnn/opt_gdn/'
    names = [('opt_gdn_chunk_cumsum.py', 'cumsum_ker'),
             ('opt_gdn_chunk_scaled_dot_kkt.py', 'kkt_ker'),
             ('opt_gdn_solve_tril.py', 'solve_tril_ker'),
             ('opt_gdn_wy_fast.py', 'wy_fast_ker'),
             ('opt_gdn_chunk_h.py', 'chunk_h_ker'),
             ('opt_gdn_chunk_o.py', 'chunk_o_ker')]
    builders, hashes = [], source_hashes(Path(__file__), Path(__file__).with_name('ascend_tla_final.py'),
                                        Path(__file__).with_name('ascend_tla_baselines.py'),
                                        LIBRARIES / 'tilelang-ascend/examples/linear_attention_and_rnn/opt_gdn_full.py')
    for filename, name in names:
        module, loaded_hashes = _load(root + filename)
        source = LIBRARIES / 'tilelang-ascend' / root / filename
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        decorator, = node.decorator_list
        options = {kw.arg: getattr(module, kw.value.id) if isinstance(kw.value, ast.Name)
                   else ast.literal_eval(kw.value) for kw in decorator.keywords}
        options['target'] = 'pto'
        builders.append(_runtime().jit(**options)(getattr(module, name).__wrapped__))
        hashes.update(loaded_hashes)
    c = 32
    kernels = [builders[0](batch, heads, length, c),
               builders[1](batch, heads, length, key_dim, c, key_dim),
               builders[2](batch, heads, length, c),
               *[builder(batch, heads, length, key_dim, value_dim, c, key_dim, value_dim)
                 for builder in builders[3:]]]
    scale = case.params.get('scale')
    scale = key_dim ** -0.5 if scale is None else scale
    identity = torch.eye(c, device=q.device, dtype=torch.float32)
    strict_mask = torch.tril(torch.ones_like(identity), diagonal=-1)
    mask = torch.tril(torch.ones_like(identity))

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        if any(x is not None for x in (initial, offsets, offsets_cpu)):
            raise UnsupportedCase('This focused opt_gdn adapter has not reviewed initial/packed state contracts.')
        qp = (q.float() * scale).to(torch.float16).transpose(1, 2).contiguous()
        kp, vp = [x.to(torch.float16).transpose(1, 2).contiguous() for x in (k, v)]
        bp = beta.to(torch.float16).transpose(1, 2).contiguous()
        gate = torch.zeros((batch, heads, length), device=q.device, dtype=torch.float32)
        workspace = torch.zeros((batch * heads, key_dim, value_dim), device=q.device, dtype=torch.float16)
        states = torch.zeros((batch, heads, length // c, key_dim, value_dim), device=q.device, dtype=torch.float16)
        # Original opt_gdn_full.py stage order and native workspace/output APIs.
        # All workspace_idx, buffering, synchronization and pass settings stay original.
        summed_gate = kernels[0](gate)
        triangular = kernels[1](kp, bp, summed_gate, strict_mask)
        triangular = kernels[2](triangular, identity)
        w, u = kernels[3](kp, vp, bp, summed_gate, triangular)
        new_v, final_state = kernels[4](kp, w, u, summed_gate, workspace, states)
        output = kernels[5](qp, kp, new_v, states, summed_gate, mask)
        return output.transpose(1, 2).contiguous().to(q.dtype), final_state.float()

    return Candidate(Implementation(run), 'tilelang-ascend::original opt_gdn_full pipeline(g=0)',
                     'Original six-stage optimized GDN composition; dense BHSD layout, native FP16 interface, query scale, zero gate and state, original32-token tile variant. Preserve every native body, pass setting, buffer and synchronization; expose both existing outputs.',
                     hashes, limitations=('Distinct original opt_gdn entry; old gdn remains quarantined. Native FP16 state must pass unchanged manifest FP32-state tolerance. No broader case expansion before this focused check.',))
