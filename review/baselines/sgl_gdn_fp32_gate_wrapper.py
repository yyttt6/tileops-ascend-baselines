"""Complete original SGL API with its documented FP32 gate-scan storage.

The original scan already loads/accumulates in FP32 and requests FP32 output.
Lossless gate promotion avoids its FP16-input/block-FP32-store compile mismatch.
Model inputs, kernel code, arithmetic precision and checker remain unchanged.
"""
import ast, dataclasses, hashlib, inspect, pathlib, textwrap


def _build(candidate, zero_initial):
    original = inspect.unwrap(candidate.implementation.run)
    assert original.__module__.endswith('.sgl.linear')
    assert candidate.api == 'sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu'
    source = textwrap.dedent(inspect.getsource(original))
    old = 'v.contiguous(),g.contiguous(),beta.contiguous(),scale='
    assert source.count(old) == 1
    source = source.replace(old, 'v.contiguous(),g.float().contiguous(),beta.contiguous(),scale=')
    if zero_initial:
        old = """if state is None:
        state=torch.zeros((cu.numel()-1,v.shape[2],q.shape[3],v.shape[3]),device=q.device,dtype=torch.float32)
    elif p.get('state_v_first'):"""
        assert source.count(old) == 1
        source = source.replace(old, "if state is not None and p.get('state_v_first'):")
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'fn']
    assert len(calls) == 1
    namespace = dict(original.__globals__)
    namespace.update({name: cell.cell_contents for name, cell in zip(original.__code__.co_freevars, original.__closure__ or (), strict=True)})
    path = pathlib.Path(__file__).resolve()
    exec(compile(source, str(path), 'exec'), namespace)
    hashes = {**candidate.source_sha256, str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    from sgl_gdn_zero_initial_wrapper import _wrap
    result = _wrap(candidate, namespace[original.__name__], hashes)
    return dataclasses.replace(result, adaptation=result.adaptation+' Timed lossless F16 gate-to-FP32 storage promotion; original gate scan already computes and requests output in FP32. '+('Use full original API zero-state mode.' if zero_initial else 'Retain original explicit FP32 zero state.'))


def instrument_original(candidate):
    return _build(candidate, False)


def specialize(case, candidate):
    assert str(case.inputs[0].dtype) == 'torch.float16' and case.inputs[5] is None
    assert not any(case.params.get(key, False) for key in ('state_v_first', 'use_gate_in_kernel', 'use_beta_sigmoid_in_kernel', 'allow_neg_eigval', 'use_qk_l2norm_in_kernel'))
    return _build(candidate, True)
