"""Editable baseline dispatch. Factories return the full callable, never outputs.

Paths are relative to this checkout. Native libraries still need their own pinned
installation; no private machine's compiled library is assumed to exist here.
"""
import importlib
import importlib.util
import os
from pathlib import Path
import sys

from review.common import HERE, REPO, read, sha


def _module(name):
    sys.path.insert(0, str(HERE / "baselines"))
    return importlib.import_module(name)


def _fla_cumsum():
    meta = read(HERE / "evidence/TOPP_FLA_CDF_ORIGINAL_SOURCE.json")
    root = Path(os.environ.get("REVIEW_FLA_ROOT", REPO / "ascend_baselines/flash-linear-attention-npu"))
    # The original source is located by repository-relative suffix, not /opt.
    suffix = str(meta["source_file"]).split("flash-linear-attention-npu/", 1)[-1]
    if suffix.startswith("/"):
        raise RuntimeError("Set REVIEW_FLA_CUMSUM_SOURCE to the pinned original FLA cumsum.py")
    source = Path(os.environ.get("REVIEW_FLA_CUMSUM_SOURCE", root / suffix))
    if sha(source) != meta["source_sha256"]:
        raise RuntimeError("FLA cumsum source differs from the original pinned API")
    spec = importlib.util.spec_from_file_location("_review_original_fla_cumsum", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_local_cumsum


def selected(case):
    """Build the paused campaign's selected candidate for an accepted case."""
    op, case_id = case._call.signature.name, case.id
    rows = read(HERE / "catalog/ACCEPTED_SELECTIONS.json")["cases"]
    matches = [r for r in rows if r["key"][1:] == [op, case_id]]
    if len(matches) != 1:
        raise ValueError("No accepted selection for this case. Use original or a custom factory.")
    row = matches[0]
    requested = getattr(case, "review_library", None)
    if requested and requested != row["key"][0]:
        raise ValueError("Selected route belongs to another library pair; use the requested original provider")
    return build_entry(case, row["candidate_entry"])


def pair(case):
    """Full4314 dispatch: never replace a library by the same case's fastest library."""
    library = getattr(case, "review_library", None)
    if not library:
        raise ValueError("Full pair tuning requires --library, e.g. torch-eager or vllm-ascend")
    rows = read(HERE / "catalog/ACCEPTED_SELECTIONS.json")["cases"]
    op = case._call.signature.name
    if any(r["key"] == [library, op, case.id] for r in rows):
        return selected(case)
    if library == "torch-eager":
        return original(case)
    os.environ["REVIEW_LIBRARY"] = library
    return vendor(case)


def best_known(case):
    """Selected accepted route, current pending route, or explicit original provider."""
    op = case._call.signature.name
    rows = read(HERE / "catalog/CASE_STATUS.json")["rows"]
    row = next(r for r in rows if (r["op"], r["case"]) == (op, case.id))
    if op == "FP8QuantPerBlockFwdOp" and row.get("candidate_route"):
        raise RuntimeError("This FP8 reference exception remains unresolved; do not infer qualification from a candidate timing")
    if row.get("candidate_route"):
        return build_entry(case, row["candidate_route"])
    if row["original_fastest_library"] == "torch-eager":
        return original(case)
    if os.environ.get("REVIEW_LIBRARY") != row["original_fastest_library"]:
        raise RuntimeError("Set REVIEW_LIBRARY=" + row["original_fastest_library"] +
                           " and prepare that original provider, or pass your custom --factory")
    return vendor(case)


def build_entry(case, entry):
    op = case._call.signature.name
    filename, symbol = entry.split(":")
    module = _module(Path(filename).stem)
    fn = getattr(module, symbol)
    if symbol in {"int4_group", "top_k_mask", "top_p_mask", "fp8_block"}:
        return fn
    if symbol == "specialize":
        raise RuntimeError("The vendor GDN specialization requires a reviewed local provider/path "
                           "binding. Its original source is included; use --factory for that binding.")
    if filename.startswith("top_p_fla_") or filename == "top_k_top_p_fla_unique_v1.py":
        return fn(cumsum=_fla_cumsum())[0]
    if filename.startswith("merge_attention_"):
        return fn(case.params, case.reference)
    if filename in {"quant_native_antiquant.py", "int8_exact_norm_floatclip.py",
                    "int8_channel_norminf_inplace.py", "int8_general_norminf.py",
                    "int8_norminf_reuse_unit.py"}:
        run = fn(op, case.params)
    else:
        run = fn(case.params)
    if run is None:
        raise RuntimeError("Selected API does not support this exact contract")
    return run


def original(case):
    """Original complete reference as a *measured* eager baseline on NPU."""
    return case.reference


def vendor(case):
    """Original registered vendor provider, with a local dependency checkout.

    Runtime/build requirements remain provider-specific. No fallback to a
    different library is performed. Use REVIEW_LIBRARY to select its label.
    """
    from benchmarks.ascend_all_pairs.registry import candidate_factory
    library = os.environ["REVIEW_LIBRARY"]
    token = importlib.import_module("benchmarks.ascend_all_pairs.registry").LIBRARY_MODULES[library]
    runtime = importlib.import_module(f"benchmarks.ascend_all_pairs.{token}.runtime")
    # Relocate only the provider's existing paths. Do not alter native sources.
    old = getattr(runtime, "WORKSPACE", None)
    if old is not None:
        for name, value in list(vars(runtime).items()):
            if isinstance(value, Path) and value.is_relative_to(old):
                setattr(runtime, name, REPO / value.relative_to(old))
    candidate = candidate_factory(case, library)()
    return candidate.implementation
