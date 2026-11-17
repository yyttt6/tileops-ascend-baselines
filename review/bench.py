#!/usr/bin/env python3
"""One local case: prepare immutable inputs, or measure one implementation.

No SSH, background dispatch, shared service, timer or project-specific card lock.
Run in the machine owner's chosen device environment. CPU commands --help and
catalog never import torch. Device commands are explicit prepare / measure.
"""
import argparse
import csv
import dataclasses
import datetime
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
import types

from review.common import HERE, REPO, TILEOPS_COMMIT, normalize_params, plain, read, sha, write_new


def install_root(root):
    root = root.resolve()
    if not (root / "benchmarks/api.py").is_file():
        raise ValueError(f"Missing full TileOPs benchmark source: {root}")
    sys.path[:0] = [str(root / "src"), str(root)]
    return root


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def move(torch, value, device):
    if isinstance(value, torch.Tensor):
        result = value.to(device)
        if result.dtype != value.dtype or result.shape != value.shape:
            raise ValueError("Transfer changed input dtype/shape")
        return result
    if isinstance(value, tuple):
        return tuple(move(torch, v, device) for v in value)
    if isinstance(value, list):
        return [move(torch, v, device) for v in value]
    if isinstance(value, dict):
        return {k: move(torch, v, device) for k, v in value.items()}
    return value


def function_identity(fn):
    return {"file": inspect.getsourcefile(fn), "text_sha256": __import__("hashlib").sha256(
        inspect.getsource(fn).encode()).hexdigest()}


def frozen_bridge(root, metadata):
    """Restore frozen numerical declarations, without patching device kernels."""
    import tileops.manifest
    # Old numerical source imports these old module names. Alias only the same
    # modern manifest classes; never rewrite an old workload/reference body.
    for old, new in [("plan", "_plan"), ("workload", "_workload")]:
        try:
            importlib.import_module("tileops.manifest." + old)
        except ModuleNotFoundError:
            sys.modules["tileops.manifest." + old] = importlib.import_module("tileops.manifest." + new)
    import workloads
    num = load_file("workloads.numerics", root / "workloads/numerics.py")
    workloads.numerics = num
    package = types.ModuleType("_review_frozen_workloads")
    package.__path__ = [str(root / "workloads")]
    sys.modules[package.__name__] = package
    relative = metadata["workload_module"].removeprefix("workloads.")
    module = importlib.import_module(package.__name__ + "." + relative)
    return getattr(module, metadata["workload_class"])


def device_setup(torch, torch_npu):
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    prop = torch.npu.get_device_properties(0)
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"hostname": os.uname().nodename, "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "cube": prop.cube_core_num,
            "vector": prop.vector_core_num, "device_name": torch.npu.get_device_name(0),
            "cann_home": os.environ.get("ASCEND_HOME_PATH"),
            "precision": "must_keep_origin_dtype; hf32 disabled",
            "process_identity": {"pid": os.getpid(), "start_ticks": stat[19],
                                 "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}}


def check_transfer_aliases(torch, values):
    """Fail closed on shared-storage views; such cases need an explicit binder."""
    tensors = []
    def walk(value):
        if isinstance(value, torch.Tensor):
            if all(value is not t for t in tensors):
                tensors.append(value)
        elif isinstance(value, (tuple, list)):
            for child in value:
                walk(child)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
    walk(values)
    stores = [(t.device, t.untyped_storage().data_ptr()) for t in tensors if t.numel()]
    if len(stores) != len(set(stores)):
        raise ValueError("Distinct input tensors share storage; provide an explicitly alias-preserving case binder")


def source_tree(root):
    return {str(p.relative_to(root)): sha(p) for folder in ["workloads", "benchmarks", "src/tileops/manifest"]
            for p in (root / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts}


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = install_root(args.source_root)
    import torch
    import torch_npu
    runtime = device_setup(torch, torch_npu)
    from benchmarks.api import cases
    from workloads.device import set_run_device
    from review.support.tools.baseline_protocol import seed_case
    set_run_device("npu")
    seed = seed_case(torch, args.op, args.case)
    case = next(c for c in cases(args.op) if c.id == args.case)
    inputs = case.inputs  # Original NPU generator, once for all later roles.
    torch.npu.synchronize()
    payload = args.output / "inputs.pt"
    torch.save(move(torch, inputs, "cpu"), payload)
    cls = type(case.workload)
    identities = source_tree(root)
    write_new(args.output / "contract.json", {
        "schema": "baseline-review-input-v1", "op": args.op, "case": args.case,
        "library": args.library,
        "seed": seed, "params": plain(case.params), "tensors": plain(case._call.tensors),
        "payload_sha256": sha(payload), "source_root": str(root),
        "workload_module": cls.__module__, "workload_class": cls.__name__,
        "reference": function_identity(case.reference),
        "verification": function_identity(case.workload.verification),
        "runtime": runtime, "source_files": identities,
        "input_identity_note": "New local original-generator payload, not claimed byte-identical to unshipped historical tensors"})
    print(f"Prepared {args.op}/{args.case}; use this exact inputs.pt for all roles")


def factory(spec, case):
    module, symbol = spec.rsplit(":", 1)
    if module.endswith(".py"):
        module = load_file("_review_custom_factory", Path(module).resolve())
    else:
        module = importlib.import_module(module)
    return getattr(module, symbol)(case)


def measure(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = read(args.inputs / "contract.json")
    if sha(args.inputs / "inputs.pt") != metadata["payload_sha256"]:
        raise ValueError("Input artifact changed")
    original_root = args.contract_root.resolve()
    for name, digest in metadata["source_files"].items():
        if sha(original_root / name) != digest:
            raise ValueError(f"Original contract source changed: {name}")
    root = install_root(args.source_root)
    if args.side == "tileops":
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if head != TILEOPS_COMMIT:
            raise ValueError(f"Expected unchanged public TileOPs {TILEOPS_COMMIT}, got {head}")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise ValueError("TileOPs tracked tree is dirty; do not change the denominator")
        frozen_cls = frozen_bridge(original_root, metadata)
    else:
        frozen_cls = None
    import torch
    import torch_npu
    runtime = device_setup(torch, torch_npu)
    from benchmarks.api import Implementation, cases
    from workloads.device import set_run_device
    from workloads.numerics import verify, Request, _snapshot, _unique, _require_unchanged
    set_run_device("cpu")
    case = next(c for c in cases(metadata["op"]) if c.id == metadata["case"])
    normalize_params(metadata["op"], case.params, metadata["params"])
    cpu_inputs = torch.load(args.inputs / "inputs.pt", map_location="cpu", weights_only=False)
    check_transfer_aliases(torch, cpu_inputs)
    case.__dict__["inputs"] = cpu_inputs
    case.__dict__["review_library"] = args.library or metadata.get("library")
    if args.library and metadata.get("library") and args.library != metadata["library"]:
        raise ValueError("Requested library differs from the prepared pair")
    if frozen_cls is not None:
        # Bind exact original methods, including their original helper globals.
        for name in ["ref_program", "verification"]:
            method = inspect.getattr_static(frozen_cls, name).__get__(case.workload, type(case.workload))
            if function_identity(method)["text_sha256"] != metadata[
                    "reference" if name == "ref_program" else "verification"]["text_sha256"]:
                raise ValueError(f"Frozen {name} binding differs; explicit family bridge required")
            if name == "ref_program":
                case.__dict__["reference"] = method
            else:
                case.__dict__["verification"] = method(*cpu_inputs)
    else:
        for name, method in [("reference", case.reference), ("verification", case.workload.verification)]:
            if function_identity(method)["text_sha256"] != metadata[name]["text_sha256"]:
                raise ValueError("Original numerical policy changed")
    device_inputs = move(torch, cpu_inputs, "npu")
    frozen_sources_before = source_tree(root)
    development_before = {str(p.relative_to(HERE)): sha(p)
                          for folder in [HERE, HERE / "baselines"] for p in folder.glob("*.py")}
    if args.factory.split(":")[0].endswith(".py"):
        custom_file = Path(args.factory.rsplit(":", 1)[0]).resolve()
        development_before[str(custom_file)] = sha(custom_file)
    # All model inputs are live arguments, never a cached result in the factory.
    if args.side == "tileops":
        import tileops.ascend
        from tileops.backend import BUILTIN, _registry
        from tileops.manifest import load_manifest
        op_name = metadata["op"]
        if "ascend" not in _registry.REGISTERED_KERNEL_TYPES.get(op_name, {}):
            raise ValueError("Public Ascend operator not registered")
        family = load_manifest()[op_name]["family"]
        cls = getattr(importlib.import_module("tileops." + family), op_name)
        op = cls(**case.arguments, target=BUILTIN)
        impl = case._entry.binder(op, case)
    else:
        value = factory(args.factory, case)
        impl = value if isinstance(value, Implementation) else Implementation(value)
    if impl.noncomparable_reason or impl.args is not None:
        raise ValueError("This generic runner requires a full comparable live-input callable; "
                         "review an explicit binder for alternate/private argument layouts")
    pristine = _snapshot(_unique(cpu_inputs))

    def check_inputs(values, before):
        _require_unchanged(_unique(move(torch, values, "cpu")), before)

    def cpu_reference(*values):
        from workloads.device import run_device
        previous = run_device()
        try:
            set_run_device("cpu")
            return case.reference(*values)
        finally:
            set_run_device(previous)

    def live_for_checker(*values):
        from workloads.device import run_device
        gpu = move(torch, values, "npu")
        before = _snapshot(_unique(values))
        previous = run_device()
        try:
            set_run_device("npu")
            if impl.reset is not None:
                impl.reset()
            produced = impl.run(*gpu)
            torch.npu.synchronize()
            check_inputs(gpu, before)
            return move(torch, produced, "cpu")
        finally:
            set_run_device(previous)

    full = verify(cpu_reference, cpu_inputs, evidence=case.verification,
                  requests={args.side: Request(live_for_checker, cpu_inputs, preserve_inputs=True)},
                  preserve_reference_inputs=True)[args.side]
    full = dataclasses.asdict(full)
    if not full["total_outputs"] or full["checked_outputs"] != full["total_outputs"] or full["unchecked_reason"]:
        raise ValueError("Full original validator did not verify every output")
    # Load exact frozen timer/parser; do not use benchmarks' CUDA/Event timer.
    tools = types.ModuleType("tools")
    tools.__path__ = [str(HERE / "support/tools")]
    sys.modules["tools"] = tools
    parser = importlib.import_module("tools.kernel_task_profile_full32")
    sys.modules["tools.kernel_task_profile"] = parser
    core = importlib.import_module("tools.measure_baselines")
    core.ROOT, core.TILEOPS_ROOT = REPO, root
    os.environ["TILEOPS_PROFILE_DIR"] = str((args.output / "profiles").resolve())
    if args.flush_factory:
        flush, buffer = factory(args.flush_factory, case)
        core._flush_kernel_cache[torch.npu.current_device()] = (flush, buffer)
    calls, last = [], []
    def observed(*values):
        result = impl.run(*values)
        torch.npu.synchronize()
        calls.append(1)
        if len(calls) == 19:
            last.append(result)
        return result
    set_run_device("npu")
    timing = core._time_reference(torch, torch_npu, observed, device_inputs, 3, 15, impl.reset)
    set_run_device("cpu")
    if len(calls) != 19 or len(last) != 1:
        raise ValueError("Expected cold1 + warm3 + formal15 actual complete API calls")
    check_inputs(device_inputs, pristine)
    measured = move(torch, last[0], "cpu")
    first = [True]
    def last_then_live(*values):
        if first[0]:
            first[0] = False
            return measured
        return live_for_checker(*values)
    check = dataclasses.asdict(verify(cpu_reference, cpu_inputs, evidence=case.verification,
        requests={args.side: Request(last_then_live, cpu_inputs, preserve_inputs=True)},
        preserve_reference_inputs=True)[args.side])
    if not check["total_outputs"] or check["checked_outputs"] != check["total_outputs"] or check["unchecked_reason"]:
        raise ValueError("Measured output failed the original full checker")
    check_inputs(device_inputs, pristine)
    if len(timing["latency_samples_ms"]) != 15 or len(timing["retained_profiles"]) != 15:
        raise ValueError("Formal run must retain exactly15 complete samples/raw windows")
    for sample, profile in zip(timing["latency_samples_ms"], timing["retained_profiles"], strict=True):
        path = Path(profile["profile_path"])
        if parser.trace_digest(path) != profile["profile_trace_sha256"]:
            raise ValueError("Raw trace SHA differs")
        piece = parser.parse_profile(path, 1)
        if not piece["profile_named_task_coverage_complete"] or abs(piece["latency_ms"] - sample) > 1e-9:
            raise ValueError("Raw trace replay disagrees with formal sample")
    maps = Path("/proc/self/maps").read_text()
    libs = {line.split()[-1] for line in maps.splitlines() if len(line.split()) >= 6
            and line.split()[-1].startswith("/") and ".so" in line.split()[-1]}
    loaded = {path: sha(path) for path in sorted(libs) if Path(path).is_file()}
    public_identity = None
    if args.side == "tileops":
        # Reuse the historical capture method on the actual executing Op;
        # do not construct another operator or compile a replacement kernel.
        from review.support.public_op_identity import FrozenQuant
        capture = FrozenQuant.__new__(FrozenQuant)
        capture.op = op
        public_identity = capture.capture(args.output)
    if source_tree(root) != frozen_sources_before:
        raise ValueError("Source tree changed during measurement")
    for path, digest in development_before.items():
        file = Path(path) if Path(path).is_absolute() else HERE / path
        if sha(file) != digest:
            raise ValueError("Factory/helper source changed during measurement")
    write_new(args.output / "result.json", {
        "schema": "baseline-review-measurement-v1", "status": "measured_pending_identity_review",
        "side": args.side, "key": [metadata["op"], metadata["case"]],
        "library": case.review_library,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device_us": statistics.mean(timing["latency_samples_ms"]) * 1000,
        "protocol": "ascend950-isolated-ffts-v3", "timing": timing,
        "full_original_check": full, "measured_output_check": check,
        "inputs_preserved": True, "payload_sha256": metadata["payload_sha256"],
        "contract_sha256": sha(args.inputs / "contract.json"),
        "runtime": runtime, "tileops_commit": TILEOPS_COMMIT if args.side == "tileops" else None,
        "factory": args.factory if args.side == "baseline" else "unchanged public Op + public full binder",
        "local_source_files": frozen_sources_before, "loaded_libraries": loaded,
        "development_sources_before_and_after": development_before,
        "tileops_generated_identity_sha256": sha(args.output / "TILEOPS_GENERATED_IDENTITY.json")
            if public_identity is not None else None,
        "review_tool_sha256": {str(p.relative_to(HERE)): sha(p) for p in HERE.glob("*.py")},
        "remaining_manual_checks": ["native/JIT executing binary identity and upstream pin",
            "same physical device/driver/firmware and approved runtime profiles",
            "actual full timed API and build flags", "no interference or profiler loss"],
        "not_published": True})
    print(f"{args.side}: {statistics.mean(timing['latency_samples_ms'])*1000:.6f} us; identity review still required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("catalog", help="CPU-only list/query; no Torch import")
    listing.add_argument("--op")
    listing.add_argument("--library")
    listing.add_argument("--fastest-subset", action="store_true", help="Historical665 subset only")
    prep = sub.add_parser("prepare", help="Original NPU input generation only; no benchmark")
    prep.add_argument("--source-root", type=Path, default=HERE / "frozen/torch20261105")
    prep.add_argument("--op", required=True)
    prep.add_argument("--library", help="Requested full pair library; required by default pair factory")
    prep.add_argument("--case", required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("measure", help="Full correctness and isolated15 device measurement")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--contract-root", type=Path, default=HERE / "frozen/torch20261105")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--side", choices=["baseline", "tileops"], required=True)
    run.add_argument("--library", help="Must match the prepared library pair")
    run.add_argument("--factory", default="review.factories:pair")
    run.add_argument("--flush-factory", help="Optional local full 1GiB flush factory(case)->(callable,buffer)")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "catalog":
        if args.fastest_subset:
            rows = read(HERE / "catalog/CASE_STATUS.json")["rows"]
        else:
            with (HERE / "catalog/ALL_OVER3_PAIRS.csv").open() as file:
                rows = list(csv.DictReader(file))
        for row in rows:
            library = row.get("library", row.get("original_fastest_library"))
            if (not args.op or row["op"] == args.op) and (not args.library or library == args.library):
                print(library, row["op"], row["case"], row.get("status", row.get("optimization_status")),
                      row.get("final_ratio", row.get("ratio")), sep="\t")
    else:
        if args.output.exists():
            raise FileExistsError("Use a new output directory; existing results are preserved")
        try:
            globals()[args.command](args)
        except BaseException:
            if args.output.exists() and args.output.is_dir():
                failure = args.output / "FAILURE.txt"
                if not failure.exists():
                    failure.write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
