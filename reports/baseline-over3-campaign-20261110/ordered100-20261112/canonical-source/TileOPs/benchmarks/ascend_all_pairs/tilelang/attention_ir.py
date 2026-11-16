"""Lossless cache of original attention PrimFuncs, with source and IR fences."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
IR_ROOT = ROOT / '.cache/all-pairs/tilelang_continuation/attention_ir001'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def _index():
    path = IR_ROOT / 'index.json'
    if not path.is_file():
        return None
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('schema') != 'tilelang-original-attention-ir/v1' or data.get('kernel_called') is not False:
        raise RuntimeError('Original attention IR cache has an invalid proof schema')
    return data, str(path.relative_to(ROOT)), _sha(raw)


@lru_cache(maxsize=1)
def _builder_ast_hash():
    path = Path(__file__).with_name('attention.py')
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == '_builder')
    return _sha(ast.dump(node, include_attributes=False).encode())


def compile_program(tilelang, builder, hashes, q_len, kv_len, head_dim, *, out_dtype, softmax_scale):
    from benchmarks.ascend_baselines import source_hashes
    hashes.update(source_hashes(Path(__file__)))
    sealed = _index()
    entry = None
    program = None
    if sealed is not None:
        index, index_path, index_hash = sealed
        parameters = {'q_len': q_len, 'head_dim': head_dim, 'out_dtype': out_dtype,
                      'softmax_scale_hex': float(softmax_scale).hex()}
        if parameters == index['fixed_parameters']:
            entry = index['entries'].get(str(kv_len))
        if entry is not None:
            if _builder_ast_hash() != index['adapter_builder_ast_sha256']:
                raise RuntimeError('Original attention builder selection changed since IR construction')
            for name, expected in index['builder_sources'].items():
                # Only the cache consumer changed; the complete _builder AST
                # and every original kernel/runtime source remain fenced.
                if name != index['adapter_path'] and hashes.get(name) != expected:
                    raise RuntimeError('Original attention IR source drift: ' + name)
            path = (ROOT / entry['ir_file']).resolve()
            if not path.is_relative_to(IR_ROOT.resolve()):
                raise RuntimeError('Original attention IR path leaves its sealed cache')
            encoded = path.read_bytes()
            if _sha(encoded) != entry['ir_sha256']:
                raise RuntimeError('Original attention IR content hash mismatch')
            program = tilelang.tvm.ir.load_json(encoded.decode())
            if _sha(program.script(show_meta=True).encode()) != entry['primfunc_script_sha256']:
                raise RuntimeError('Restored original PrimFunc differs from the CPU proof')
            hashes[index_path] = index_hash
            hashes[str(path.relative_to(ROOT))] = entry['ir_sha256']
    if program is None:
        program = builder(q_len, kv_len, head_dim, out_dtype=out_dtype, softmax_scale=softmax_scale)
    kernel = tilelang.compile(program, target='ascend', out_idx=-1)
    if entry is not None:
        if _sha(kernel.get_kernel_source().encode()) != entry['kernel_source_sha256']:
            raise RuntimeError('Compiled source differs from the original CPU compilation')
        if getattr(kernel, '_tilelang_cache_key', None) != entry['cache_key']:
            raise RuntimeError('Restored original PrimFunc changed the original JIT cache key')
    return kernel
