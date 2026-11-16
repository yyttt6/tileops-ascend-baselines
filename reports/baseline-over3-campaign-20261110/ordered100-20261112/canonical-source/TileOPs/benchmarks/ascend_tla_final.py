"""Original TLA normalization, routing and GDN with host interface adaptations."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
from benchmarks.ascend_tla_baselines import _runtime


def _load(relative, *, transform=None):
    _runtime()
    original = LIBRARIES / "tilelang-ascend" / relative
    tree = ast.parse(original.read_text(), str(original))
    # Skip standalone demo execution while keeping declarations and constants.
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")]
    if transform:
        tree = transform(tree)
    text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    key = hashlib.sha256(text.encode()).hexdigest()[:20]
    cache = WORKSPACE / ".cache/ascend-classic" / ("tla-final-" + key)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / original.name
    path.write_text(text)
    sys.path.insert(0, str(original.parent))
    name = "_classic_tla_final_" + key
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, source_hashes(original, path)


def tla_rms(case):
    import torch
    from benchmarks.api import Implementation

    x, weight = case.inputs
    rows, hidden = x.shape
    module, hashes = _load("examples/xllm_kernels/split_qkv_rmsnorm_mrope.py")
    module.COMPILE_MAX_TOKENS = rows
    program = module.build_split_qkv_rmsnorm_mrope_kernel(
        head_size=hidden, rope_dim=64, num_q_heads=1, num_kv_heads=1,
        vec_core_num=module.detect_vec_core_num(), max_num_tokens=rows)
    kernel = _runtime().compile(program, target="pto", pass_configs=module.KERNEL_PASS_CONFIGS)
    pattern = module.build_mrope_gather_pattern_merged(
        half_rope_dim=32, rope_dim=64, mrope_section=(11, 11, 10),
        is_interleaved=False, gather_pad_dim=256, device=str(x.device))

    def run(x, weight):
        packed = x[:, None, :].expand(rows, 4, hidden).contiguous()
        phase = torch.zeros((rows, 3, 64), device=x.device, dtype=x.dtype)
        phase[..., :32] = 1
        outputs = [torch.empty((rows, 1, hidden), device=x.device, dtype=x.dtype) for _ in range(4)]
        gamma = weight.reshape(1, hidden).contiguous()
        kernel(packed, gamma, gamma, phase.reshape(rows, 192), pattern,
               *outputs, rows, case.params["eps"])
        return outputs[0].reshape(x.shape)

    return Candidate(
        Implementation(run), "tilelang-ascend::split_qkv_rmsnorm_mrope Q output",
        "One head4096, original epsilon/gamma, identity rotary cos1/sin0, select Q; original schedule, existing PTO backend.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang-ascend/examples/xllm_kernels/utils.py") | hashes,
    )


def tla_prepermute(case):
    import torch
    from benchmarks.api import Implementation

    def expose_histogram(tree):
        changed = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build":
                call, = node.decorator_list
                for kw in call.keywords:
                    if kw.arg == "out_idx":
                        assert ast.literal_eval(kw.value) == [2, 3]
                        kw.value = ast.List(elts=[ast.Constant(i) for i in (2, 3, 4)], ctx=ast.Load())
                    if kw.arg == "workspace_idx":
                        assert ast.literal_eval(kw.value) == [4]
                        kw.value = ast.List(elts=[], ctx=ast.Load())
                changed += 1
        assert changed == 1
        return tree

    module, hashes = _load("examples/moe_token_permute/moe_token_permute.py",
                           transform=expose_histogram)
    x, ids = case.inputs
    rows, hidden = x.shape
    topk = ids.shape[-1]
    experts = case.params["num_local_experts"]
    kernel, padded = module._compile_fused(rows, topk, hidden, ids.numel(), ids.numel(),
                                           experts, dtype=str(x.dtype).split(".")[-1])

    def run(x, ids):
        indices = torch.zeros((1, padded), device=x.device, dtype=torch.int32)
        indices[0, :ids.numel()].copy_(ids.reshape(-1))
        out, inverse, histogram = kernel(x.contiguous(), indices)
        # Format native per-core counts as the required cumulative expert ends.
        ends = histogram.reshape(-1, experts).sum(0).cumsum(0).to(torch.int32)
        return out, ends, inverse.reshape(-1)[:ids.numel()].contiguous()

    return Candidate(
        Implementation(run), "tilelang-ascend::moe_token_permute",
        "Original routing kernel; expose existing histogram workspace as output and format it as prefix counts. "
        "Kernel body and synchronization unchanged.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang-ascend/examples/moe_token_permute/moe_token_utils.py") | hashes,
    )


def tla_deltanet(case):
    import torch
    from benchmarks.api import Implementation

    if any(t is not None for t in case.inputs[4:]):
        raise UnsupportedCase("Original sample uses zero initial state and dense batches.")
    def expose_final(tree):
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_chunk_gdn"]
        assert isinstance(fn.body[-1], ast.Return) and isinstance(fn.body[-1].value, ast.Name)
        assert fn.body[-1].value.id == "o"
        fn.body[-1].value = ast.Tuple(elts=[ast.Name(id="o", ctx=ast.Load()),
                                          ast.Name(id="fs", ctx=ast.Load())], ctx=ast.Load())
        return tree
    module, hashes = _load("examples/linear_attention_and_rnn/gdn_full.py",
                           transform=expose_final)
    # Select the library's existing PTO backend for its tile intrinsics.
    # Preserve every original kernel body and every launch/pass option.
    deps = list((LIBRARIES / "tilelang-ascend/examples/linear_attention_and_rnn/gdn").glob("*.py"))
    for source in deps:
        imported = sys.modules.get("gdn." + source.stem)
        if imported is None:
            continue
        for node in ast.parse(source.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "tilelang" and decorator.func.attr == "jit"):
                    options = {}
                    for kw in decorator.keywords:
                        assert kw.arg is not None
                        options[kw.arg] = (getattr(imported, kw.value.id)
                                           if isinstance(kw.value, ast.Name)
                                           else ast.literal_eval(kw.value))
                    options["target"] = "pto"
                    original = getattr(imported, node.name)
                    setattr(imported, node.name, _runtime().jit(**options)(original.__wrapped__))

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        scale = case.params.get("scale")
        if scale is None:
            scale = q.shape[-1] ** -0.5
        qp = (q.float() * scale).to(torch.float16).transpose(1, 2).contiguous()
        kp, vp = [t.to(torch.float16).transpose(1, 2).contiguous() for t in (k, v)]
        bp = beta.to(torch.float16).transpose(1, 2).contiguous()
        gate = torch.zeros(bp.shape, device=q.device, dtype=torch.float32)
        out, final = module.kernel_chunk_gdn(qp, kp, vp, gate, bp, 32, 32, 32)
        return out.transpose(1, 2).contiguous().to(q.dtype), final.float()

    return Candidate(
        Implementation(run), "tilelang-ascend::kernel_chunk_gdn(g=0)",
        "Original full32-chunk GDN with zero log gate; original FP16 API, BSHD/BHSD transpose, expose computed final state; existing PTO backend.",
        source_hashes(Path(__file__), *deps) | hashes,
        limitations=("Original state is stored in FP16; casting to FP32 does not restore lost accuracy, verification required.",),
    )
