"""Exact tensor lerp: skip only two zero-tail pads in original affine helper."""
import ast
import copy
import dataclasses
import hashlib
import pathlib


def clone_affine(module):
    source = pathlib.Path(module.__file__).read_text()
    original = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_affine')
    expected = ast.dump(original, include_attributes=False)
    node = copy.deepcopy(original)

    class Forward(ast.NodeTransformer):
        count = 0

        def visit_Call(self, n):
            n = self.generic_visit(n)
            if (isinstance(n.func, ast.Attribute) and
                    isinstance(n.func.value, ast.Name) and
                    n.func.value.id == 'F' and n.func.attr == 'pad'):
                assert ast.unparse(n.args[1]) == '(0, tail)' and not n.keywords
                self.count += 1
                return ast.IfExp(test=ast.Name(id='tail', ctx=ast.Load()),
                                 body=n, orelse=copy.deepcopy(n.args[0]))
            return n

    forward = Forward()
    node = forward.visit(node)
    assert forward.count == 2

    class Reverse(ast.NodeTransformer):
        def visit_IfExp(self, n):
            if isinstance(n.test, ast.Name) and n.test.id == 'tail':
                assert ast.dump(n.orelse) == ast.dump(n.body.args[0])
                return copy.deepcopy(n.body)
            return self.generic_visit(n)

    assert ast.dump(Reverse().visit(copy.deepcopy(node)), include_attributes=False) == expected
    env = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 module.__file__, 'exec'), env)
    return env['_affine'], {'reverse_affine_AST_identical': True,
                            'only_change': 'skip F.pad when original tail equals zero',
                            'pad_sites': 2}


def build(case, library):
    assert library == 'sgl-kernel-npu' and case._call.signature.name == 'LerpTensorFwdOp'
    from benchmarks.ascend_all_pairs.sgl import extra_arithmetic
    affine, proof = clone_affine(extra_arithmetic)
    original = next(n for n in ast.parse(pathlib.Path(extra_arithmetic.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'lerp')
    env = dict(vars(extra_arithmetic))
    env['_affine'] = affine
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
                 extra_arithmetic.__file__, 'exec'), env)
    candidate = env['lerp'](case)
    path = pathlib.Path(__file__)
    candidate.source_sha256[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dataclasses.replace(candidate, api=candidate.api + ' (affine zero-tail pad skip)',
                               adaptation='Only zero-tail affine pads skipped; original complete lerp timed body, FP32 conversions, product, affine calls and output cast unchanged.')
