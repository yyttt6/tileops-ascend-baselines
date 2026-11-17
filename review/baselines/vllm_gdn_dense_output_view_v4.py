"""Host-wrapper specialization of an existing complete vLLM KDA/GDN candidate.

No kernel, compiler flag, precision, checker, or model input is changed. The
original provider callable and source identity must be validated before use.
"""
import ast
import json
import dataclasses
import hashlib
import inspect
import pathlib
import textwrap


def transformed_source(original):
    source=textwrap.dedent(inspect.getsource(original))
    changes=[
        ("q, k = (F.pad(x, (0, dim-kd)) for x in (q, k))", "if dim != kd:\n        q, k = (F.pad(x, (0, dim-kd)) for x in (q, k))"),
        ("v = F.pad(v, (0, dim-vd))", "if dim != vd:\n        v = F.pad(v, (0, dim-vd))"),
        ("boundaries = (torch.arange(b + 1, device=q.device, dtype=torch.int64) * t if supplied is None\n                  else supplied.to(device=q.device, dtype=torch.int64).clone())", "dense_decode = t == 1 and supplied is None\n    boundaries = (None if dense_decode else (torch.arange(b + 1, device=q.device, dtype=torch.int64) * t if supplied is None\n                  else supplied.to(device=q.device, dtype=torch.int64).clone()))"),
        ("sequences = boundaries.numel() - 1", "sequences = b if dense_decode else boundaries.numel() - 1"),
        ("state = F.pad(state.float(), (0, dim-kd, 0, dim-vd)).clone().contiguous()", "state = state.float()\n        if dim != kd or dim != vd:\n            state = F.pad(state, (0, dim-kd, 0, dim-vd))\n        state = state.contiguous()"),
        ("qp, kp = (x.reshape(1, b*t, padded_heads, dim).contiguous() for x in (q, k))\n    vp = v.reshape(1, b*t, padded_heads, dim).clone().contiguous()\n    gp = gate.reshape(1, b*t, padded_heads, dim).contiguous()\n    bp = beta.reshape(1, b*t, padded_heads).contiguous()", "leading = (b, t) if dense_decode else (1, b*t)\n    qp, kp = (x.reshape(*leading, padded_heads, dim).contiguous() for x in (q, k))\n    vp = v.reshape(*leading, padded_heads, dim)\n    vp = vp.contiguous() if dense_decode else vp.clone().contiguous()\n    gp = gate.reshape(*leading, padded_heads, dim).contiguous()\n    bp = beta.reshape(*leading, padded_heads).contiguous()"),
    ]
    for old,new in changes:
        assert source.count(old)==1,('original host wrapper changed; no blind rewrite',old)
        source=source.replace(old,new)
    shape_line='b, t, qh, kd = q.shape'
    assert source.count(shape_line)==1
    source=source.replace(shape_line,shape_line+'\n    assert b == 1 and t == 1, \"This out-of-place native route is qualified only for B1T1\"')
    final_line='final = final.transpose(-1, -2).contiguous()'
    assert source.count(final_line)==1
    source=source.replace(final_line,'final = final.transpose(-1, -2)')
    tree=ast.parse(source)
    # Complete original APIs remain the same; only their dense private-state
    # mode and surrounding identity/layout operations change.
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native']
    assert calls.count('fused_recurrent_kda')==calls.count('chunk_kda')==1
    return source


def specialize(candidate):
    review_path=pathlib.Path(__file__).resolve().parent/'VLLM_GDN_OUTPUT_VIEW_CONTRACT_REVIEW.json'
    review=json.loads(review_path.read_text())
    assert review['cases']==12 and review['no_contiguous_output_requirement']
    assert all(not value for row in review['rows'] for value in row['output_contiguous_requirements'].values())
    assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==digest for p,digest in review['sources'].items())
    original=inspect.unwrap(candidate.implementation.run)
    assert original.__name__=='run' and original.__module__.endswith('.vllm.linear_attention'),'complete original host wrapper required'
    namespace=dict(original.__globals__)
    values, empty = {}, []
    for name, cell in zip(original.__code__.co_freevars, original.__closure__ or (), strict=True):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            empty.append(name)
    namespace.update(values)
    # Original provider creates these cells only for enabled gate branches.
    # Preserve their absence for disabled branches; never fabricate a module.
    for name in empty:
        assert name in ('gating', 'multiply'), ('unexpected empty closure', name)
        if name == 'gating':
            assert not namespace['raw'] and not namespace['beta_sigmoid']
        else:
            assert not (namespace['p'].get('allow_neg_eigval', False) and namespace['beta_sigmoid'])
        namespace.pop(name, None)
    source=transformed_source(original)
    exec(compile(source,str(pathlib.Path(__file__).resolve()),'exec'),namespace)
    path=pathlib.Path(__file__).resolve()
    hashes={**candidate.source_sha256,str(path):hashlib.sha256(path.read_bytes()).hexdigest(),str(review_path):hashlib.sha256(review_path.read_bytes()).hexdigest(),**review['sources']}
    from benchmarks.ascend_all_pairs.jit_evidence import triton_run
    from benchmarks.ascend_all_pairs.vllm.runtime import CACHE
    measured=triton_run(namespace[original.__name__],hashes,CACHE)
    return dataclasses.replace(candidate,
        implementation=dataclasses.replace(candidate.implementation,run=measured),
        source_sha256=hashes,
        adaptation=candidate.adaptation+' Wrapper specialization: omit exact zero padding; dense T1 uses original recurrent API without synthetic sequence boundaries and with the original out-of-place FP32 final-state mode; original value inputs remain read-only; keep all declared outputs and input layout transforms timed; final FP32 logical-state transpose returns a view because frozen signature requires no contiguous output. B1T1 only; no kernel or precision changes.')
