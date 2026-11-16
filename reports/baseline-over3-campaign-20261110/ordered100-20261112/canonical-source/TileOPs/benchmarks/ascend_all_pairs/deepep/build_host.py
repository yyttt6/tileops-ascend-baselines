"""CPU-only host syntax/build driver; never imports torch, torch_npu or vendor.

Default: syntax-check the real pybind translation unit without linking/loading.
--build --cxx11-abi N: build an original-source host extension, once the host
compiler and dependencies are available. This still does not load the library
or compile/execute device kernels. ABI must be supplied from the target Torch
build; we never import Torch to guess it.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone

if __package__:
    from .runtime import CACHE, VENDOR, WORKSPACE, digest, source_snapshot
else:
    from runtime import CACHE, VENDOR, WORKSPACE, digest, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1))
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--site-packages", type=Path, default=Path("/opt/tileops-ci/runtime/lib/python3.12/site-packages"))
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("deployment.json"))
    args = parser.parse_args()
    if args.build and args.cxx11_abi is None:
        parser.error("--build requires --cxx11-abi from the target Torch build")
    build = CACHE / "host"
    build.mkdir(parents=True, exist_ok=True)
    torch = args.site_packages / "torch"
    npu = args.site_packages / "torch_npu"
    arch = platform.machine()
    if arch not in ("aarch64", "x86_64"):
        parser.error(f"Unsupported host architecture: {arch}")
    includes = [VENDOR / "csrc", VENDOR / "deep_ep/include", VENDOR / "third-party/deep_jit/include",
                args.cann / f"{arch}-linux/include", args.cann / f"{arch}-linux/asc",
                args.cann / f"{arch}-linux/asc/include", npu / "include",
                npu / "include/third_party/acl/inc", torch / "include",
                torch / "include/torch/csrc/api/include", Path(sysconfig.get_paths()["include"])]
    cmd = [args.compiler, "-std=c++20", "-fPIC", "-DTORCH_EXTENSION_NAME=_C",
           "-Wno-deprecated-declarations", "-Wno-attributes"]
    cmd += ["-I" + str(p) for p in includes]
    if args.cxx11_abi is not None:
        cmd.append(f"-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}")
    source = VENDOR / "csrc/python_api.cpp"
    output = build / "lib/deep_ep" / ("_C" + sysconfig.get_config_var("EXT_SUFFIX"))
    if args.build:
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-O3", "-shared", str(source), "-o", str(output)]
        for lib in (args.cann / "lib64", npu / "lib", torch / "lib"):
            cmd += ["-L" + str(lib), "-Wl,-rpath," + str(lib)]
        cmd += ["-Wl,-Bsymbolic-functions", "-Wl,--enable-new-dtags",
                "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10",
                "-lascendcl", "-lhccl", "-lhcomm", "-ltorch_npu"]
    else:
        cmd += ["-fsyntax-only", str(source)]
    record = {"schema": "deepep-host-deployment-v1", "architecture": platform.machine(),
              "python_abi": sys.implementation.cache_tag, "cann_path": str(args.cann.resolve()),
              "command": cmd, "sources": source_snapshot(), "device_execution": False,
              "timestamp": datetime.now(timezone.utc).isoformat(), "mode": "build" if args.build else "syntax_only"}
    try:
        result = subprocess.run(cmd, cwd=build, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        log, code = result.stdout, result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        log, code = str(exc), 124
    log_path = build / ("build.log" if args.build else "syntax.log")
    log_path.write_text(log)
    record.update(returncode=code, log_path=str(log_path.relative_to(WORKSPACE)),
                  status=("built" if args.build else "syntax_pass") if code == 0 else "deployment_blocked")
    if code == 0 and args.build:
        # Copy only source/header package files, never any vendor .so or cache.
        for path in (VENDOR / "deep_ep").rglob("*"):
            if path.is_file() and path.suffix in (".py", ".hpp"):
                dest = output.parent / path.relative_to(VENDOR / "deep_ep")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        record["binary_sha256"] = digest(output)
        (build / "lib/build_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "returncode": code, "log_path": str(log_path)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
