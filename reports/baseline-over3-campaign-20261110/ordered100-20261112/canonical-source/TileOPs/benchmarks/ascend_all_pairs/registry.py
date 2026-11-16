"""Resolve a single library without importing other libraries' runtimes.

Each library owns one package exporting ``PROVIDERS[op](case) -> Candidate``.
Imports are lazy so merely reading this registry does not initialize a device.
The historical classic providers remain available while libraries are extended.
"""

from importlib import import_module
from pathlib import Path


LIBRARY_MODULES = {
    "ops-math": "ops_math",
    "ops-nn": "ops_nn",
    "ops-transformer": "ops_transformer",
    "sgl-kernel-npu": "sgl",
    "flash-linear-attention-npu": "fla",
    "TileKernels": "tilekernels",
    "tilelang": "tilelang",
    "ops-tensor": "ops_tensor",
    "catlass": "catlass",
    "DeepGEMM-Ascend": "deepgemm",
    "DeepEP-Ascend": "deepep",
    "DeepSelect": "deepselect",
    "FlashMLA": "flashmla",
    "vllm-ascend": "vllm",
    "tilelang-ascend": "tla",
    "atvc": "atvc",
    "ops-fft": "ops_fft",
}


def candidate_factory(case, library):
    """Return the selected factory; never substitute another library on error."""
    if library not in LIBRARY_MODULES:
        raise ValueError(f"Unknown Ascend baseline library: {library}")
    token = LIBRARY_MODULES[library]
    package = Path(__file__).parent / token / "__init__.py"
    op = case._call.signature.name
    if package.is_file():
        module = import_module(f"benchmarks.ascend_all_pairs.{token}")
        providers = module.PROVIDERS
        if not isinstance(providers, dict) or not all(
            isinstance(name, str) and callable(factory) for name, factory in providers.items()
        ):
            raise TypeError(f"{module.__name__}.PROVIDERS must map op names to factories")
        if op in providers:
            return lambda: providers[op](case)

    # Keep the exact historical factory and its restrictions for representatives.
    # A failure above is intentionally not caught or replaced by this path.
    from benchmarks.ops.bench_ascend_classics import candidate_factories

    factories = candidate_factories(case)
    if library not in factories:
        from benchmarks.ascend_baselines import UnsupportedCase

        raise UnsupportedCase(f"No registered provider for {op} / {library}")
    return factories[library]
