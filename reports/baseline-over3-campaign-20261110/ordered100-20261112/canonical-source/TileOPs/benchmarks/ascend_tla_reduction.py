"""Shape-only adaptation of the original PTO row maximum example."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
from benchmarks.ascend_tla_baselines import _runtime


def tla_amax(case):
    import torch
    from benchmarks.api import Implementation

    _runtime()
    x, = case.inputs
    if x.ndim != 2 or case.params["dim"] != -1 or case.params["keepdim"]:
        raise UnsupportedCase("This adapter requires complete reduction rows.")
    source = LIBRARIES / "tilelang-ascend/examples/reduce/example_row_reduce_max_slice_buffer.py"
    text = source.read_text()
    width = x.shape[-1]
    # Choose only a batch-row extent; never split the reduction or change the
    # original kernel's loops, instructions, pass configuration, or schedule.
    row_tile = next((r for r in (4, 2, 1) if (r * width + 8) * 4 <= 208 * 1024), None)
    if row_tile is None:
        raise UnsupportedCase('One complete original FP32 reduction row exceeds its UB allocation.')
    replacements = {
        "T.Tensor([4, 8], dtype)": f"T.Tensor([{row_tile}, {width}], dtype)",
        "T.alloc_ub((4, 8), dtype)": f"T.alloc_ub(({row_tile}, {width}), dtype)",
        "real_shape=[4, 4]": f"real_shape=[{row_tile}, {width}]",
        "T.Tensor([1, 8], dtype)": f"T.Tensor([{row_tile}, 1], dtype)",
        "T.alloc_ub((1, 8), dtype=dtype)": f"T.alloc_ub(({row_tile}, 1), dtype=dtype)",
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise RuntimeError(f"Original reduction source changed: {before}")
        text = text.replace(before, after)
    original = ast.parse(source.read_text())
    adapted = ast.parse(text)

    class EraseConstants(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(None), node)

    # Every changed AST node is a dimension constant; no loop, flag, layout,
    # intrinsic, condition or scheduling annotation may change.
    if ast.dump(EraseConstants().visit(original)) != ast.dump(EraseConstants().visit(ast.parse(text))):
        raise RuntimeError("Reduction adaptation changed more than dimensions.")
    adapted.body = [
        node for node in adapted.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (isinstance(node, ast.Assign) and
            all(isinstance(t, ast.Name) and t.id == "pass_configs" for t in node.targets))
    ]
    generated = ast.unparse(adapted) + "\n"
    key = hashlib.sha256(generated.encode()).hexdigest()[:20]
    cache = WORKSPACE / ".cache/ascend-classic" / f"tla-reduce-{key}"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "original_with_manifest_width.py"
    path.write_text(generated)
    name = "_classic_tla_reduce_" + key
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    kernel = module.reduce_max_slice_buffer()

    def run(x):
        packed = x.float().contiguous()
        original_rows = packed.shape[0]
        if original_rows % row_tile:
            packed = torch.nn.functional.pad(packed, (0, 0, 0, -original_rows % row_tile))
        outputs = [kernel(chunk).reshape(row_tile)
                   for chunk in packed.split(row_tile, dim=0)]
        return torch.cat(outputs)[:original_rows].to(x.dtype)

    return Candidate(
        Implementation(run), "tilelang-ascend::reduce_max_slice_buffer",
        "Only input row/width and real_shape constants changed; matching native output dimensions; exact FP32 widening, independent-row padding and crop.",
        source_hashes(Path(__file__), source, path,
                      Path(__file__).with_name("ascend_tla_baselines.py")),
        limitations=("Host launch loop and all conversions count toward end-to-end execution.",),
    )
