"""Compile the original sources without importing torch/torch_npu or loading ACL.

Run with Python -I -S. This is a compiler/linker process, never a device test.
All artifacts stay under .cache/all-pairs/deepselect; vendor sources are read-only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
CACHE = WORKSPACE / ".cache/all-pairs/deepselect"
VENDOR = WORKSPACE / "ascend_baselines/DeepSelect"


def prepare_api(destination):
    """Change only exception-message formatting in a private host-source copy.

    CANN 9.1's C++ library does not expose std::format. TORCH_CHECK already
    supports concatenated arguments. Selection, dispatch and checks are identical.
    """
    old = '''std::format("{}.stride(0) (currently {} numbers) must be a multiple of {} Bytes ({} numbers)",
                tensor_name, cur_stride,
                alignment_requirement_bytes, alignment_requirement_bytes / itemsize
            )'''
    new = '''tensor_name, ".stride(0) (currently ", cur_stride,
            " numbers) must be a multiple of ", alignment_requirement_bytes,
            " Bytes (", alignment_requirement_bytes / itemsize, " numbers)"'''
    original = (VENDOR / "csrc/api.cpp").read_text()
    if original.count(old) != 1:
        raise RuntimeError("Original API diagnostic changed; review the compatibility replacement before building.")
    destination.write_text(original.replace(old, new))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--python", type=Path, default=Path("/opt/tileops-ci/runtime/bin/python"))
    parser.add_argument("--site-packages", type=Path, default=Path("/opt/tileops-ci/runtime/lib/python3.12/site-packages"))
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--gcc-toolchain", type=Path,
                        help="Optional compatible GCC headers/runtime for bisheng")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = CACHE / "builds" / stamp
    run_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(run_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    torch = args.site_packages / "torch"
    torch_npu = args.site_packages / "torch_npu"
    cann = args.cann.resolve()
    machine = platform.machine()
    if machine not in ("aarch64", "x86_64"):
        raise RuntimeError(f"Unsupported build host architecture: {machine}")
    cann_arch = cann / (machine + "-linux")
    compiler = cann / "tools/bisheng_compiler/bin/bisheng"
    config = json.loads(subprocess.check_output([
        str(args.python), "-I", "-S", "-c",
        "import json,sysconfig; print(json.dumps({'include':sysconfig.get_paths()['include'],'suffix':sysconfig.get_config_var('EXT_SUFFIX')}))",
    ], text=True))
    abi = args.cxx11_abi
    if abi is None:
        # Read symbol names; do NOT load the library to query its ABI.
        symbols = subprocess.check_output(["nm", "-D", str(torch / "lib/libc10.so")], text=True)
        asserted = [line for line in symbols.splitlines() if "torchInternalAssertFail" in line and "basic_string" in line]
        if not asserted:
            raise RuntimeError("Cannot infer libc10 string ABI from symbols; supply --cxx11-abi explicitly.")
        abi = int(any("__cxx11" in line for line in asserted))
    common = [str(compiler), "-O2", "-std=c++20", "--npu-arch=dav-3510", "-fPIC",
              "-DDEEP_SELECT_IS_BUILD_ON_ASCEND", "-Wno-deprecated-declarations",
              f"-D_GLIBCXX_USE_CXX11_ABI={abi}", "-I" + str(VENDOR / "csrc"),
              "-I" + str(VENDOR / "csrc/3rdparty/kerutils/include"), "-I" + str(cann / "include"),
              "-isystem", str(cann_arch / "asc/include"),
              "-isystem", str(torch_npu / "include"),
              "-isystem", str(torch_npu / "include/third_party/acl/inc")]
    if args.gcc_toolchain is not None:
        common.append("--gcc-toolchain=" + str(args.gcc_toolchain.resolve()))
    adapted_api = prepare_api(run_dir / "api_compat.cpp")
    commands = {
        "kernel": common + ["-mllvm", "-cce-aicore-dcpreload-args=false", "-c", str(VENDOR / "csrc/ascend_kernels/kernel.asc"), "-o", str(run_dir / "kernel.o")],
        "api": common + ["-DTORCH_EXTENSION_NAME=deep_select_npu", "-DTORCH_API_INCLUDE_EXTENSION_H",
                         "-I" + str(torch / "include"), "-I" + str(torch / "include/torch/csrc/api/include"),
                         "-I" + config["include"], "-c", str(adapted_api), "-o", str(run_dir / "api.o")],
    }
    report = {"schema": "deepselect-cpu-build/v1", "device_initialized": False,
              "machine": platform.machine(), "cann": str(cann), "cxx11_abi": abi,
              "vendor_modified": False, "commands": [], "status": "building"}

    def execute(stage, command):
        try:
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
            code, output = result.returncode, result.stdout
        except subprocess.TimeoutExpired as error:
            code, output = 124, str(error)
        log = run_dir / f"{stage}.log"
        log.write_text(output)
        report["commands"].append({"stage": stage, "argv": command, "exit_code": code, "log": str(log.relative_to(WORKSPACE))})
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return code

    failed = [stage for stage, command in commands.items() if execute(stage, command)]
    if failed:
        report.update(status="deployment_blocked", failed_stages=failed)
    else:
        build = CACHE / "native"
        package = build / "deep_select"
        package.mkdir(parents=True, exist_ok=True)
        extension = package / ("deep_select_npu" + config["suffix"])
        command = [str(compiler), "-shared", str(run_dir / "api.o"), str(run_dir / "kernel.o"),
                   "-L" + str(torch / "lib"), "-L" + str(torch_npu / "lib"), "-L" + str(cann / "lib64"),
                   "-Wl,-Bsymbolic-functions", "-ltorch_python", "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu", "-lascendcl",
                   str(cann_arch / "lib64/libascendc_runtime.a"),
                   "-Wl,-rpath," + str(torch / "lib"), "-Wl,-rpath," + str(torch_npu / "lib"),
                   "-Wl,-rpath," + str(cann / "lib64"), "-o", str(extension)]
        if execute("link", command):
            report.update(status="deployment_blocked", failed_stages=["link"])
        else:
            for name in ("__init__.py", "__version__.py", "interface.py"):
                shutil.copyfile(VENDOR / "deep_select" / name, package / name)
            spec = importlib.util.spec_from_file_location("deepselect_build_metadata", HERE / "loader.py")
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            receipt = {"machine": platform.machine(), "source_sha256": loader.vendor_hashes(),
                       "extension": str(extension.relative_to(build)), "extension_sha256": loader.digest(extension),
                       "compiler_report": str((run_dir / "report.json").relative_to(WORKSPACE)),
                       "compatibility_sha256": {str(adapted_api.relative_to(WORKSPACE)): loader.digest(adapted_api),
                                                str(Path(__file__).relative_to(WORKSPACE)): loader.digest(__file__)},
                       "host_compatibility": "TORCH_CHECK diagnostic formatting only; no math or schedule change.",
                       "device_validation": "not_run"}
            (build / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            report["status"] = "compiled_unverified"
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(run_dir / "report.json")}))
    return 0 if report["status"] == "compiled_unverified" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
