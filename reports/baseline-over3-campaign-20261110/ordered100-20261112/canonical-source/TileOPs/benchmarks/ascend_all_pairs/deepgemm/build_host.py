"""CPU-only deployment probe / host extension builder, using no torch imports.

Run with the desired Python interpreter. The default only probes prerequisites.
An explicit --build --cxx11-abi {0,1} builds the unchanged C++ host extension.
No extension import, linking test executable, device compiler, or NPU query runs.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

if __package__:
    from .runtime import CACHE, VENDOR, WORKSPACE, fingerprint, sha256
else:
    from runtime import CACHE, VENDOR, WORKSPACE, fingerprint, sha256


def package_path(name):
    try:
        return Path(importlib.metadata.distribution(name).locate_file(name)).resolve()
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("deployment_probe.json"))
    args = parser.parse_args()
    if args.build and args.cxx11_abi is None:
        parser.error("--build requires --cxx11-abi matching the selected torch distribution")
    CACHE.mkdir(parents=True, exist_ok=True)
    temp = CACHE / "host_build"
    temp.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(temp)
    torch_dir, npu_dir, tilelang_dir = (package_path(x) for x in ("torch", "torch_npu", "tilelang"))
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"))).resolve()
    compiler = shutil.which(args.cxx)
    report = {
        "schema": "deepgemm-host-deployment/v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": platform.machine(), "python": sys.executable,
        "python_cache_tag": sys.implementation.cache_tag,
        "toolkit": str(toolkit), "compiler": compiler,
        "imports": "stdlib and own runtime only; no torch, torch_npu, ACL, tilelang or vendor import",
        "device_initialized": False, "host_build_attempted": args.build,
        "blockers": [], "checks": [],
    }
    info = toolkit / "share/info/runtime/version.info"
    if info.is_file():
        report["runtime_version_info"] = {"path": str(info), "sha256": sha256(info), "text": info.read_text()}
    if compiler is None:
        report["blockers"].append(f"Compiler not found: {args.cxx}")
    else:
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True, env=env)
        report["compiler_version"] = version.stdout.splitlines()[0]
        probe = temp / "format_probe.cpp"
        probe.write_text('#include <format>\nint main() { return std::format("{}", 1).size() != 1; }\n')
        command = [compiler, "-std=c++20", "-fsyntax-only", str(probe)]
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        report["checks"].append({"name": "cxx20_format", "command": command, "returncode": result.returncode, "stderr": result.stderr})
        if result.returncode:
            report["blockers"].append("Selected compiler cannot compile C++20 <format>, required by csrc/utils/format.hpp and DeepJIT headers")
    required = {
        "python_header": Path(sysconfig.get_paths()["include"]) / "Python.h",
        "acl_header": toolkit / "include/acl/acl.h",
        "hccl_header": toolkit / "include/hccl/hccl.h",
        "bisheng_for_future_device_jit": (
            toolkit / "bin/bisheng" if (toolkit / "bin/bisheng").is_file()
            else Path(shutil.which("bisheng")) if shutil.which("bisheng") else None
        ),
        "torch_header": torch_dir / "include/torch/extension.h" if torch_dir else None,
        "torch_npu_header": npu_dir / "include/torch_npu/csrc/core/npu/NPUStream.h" if npu_dir else None,
        "tilelang_package_for_vendor_init": tilelang_dir / "__init__.py" if tilelang_dir else None,
        "deepjit_header": VENDOR / "third-party/deep-jit/include/deep_jit/python_api.hpp",
        "magic_enum_header": VENDOR / "third-party/magic_enum/include/magic_enum/magic_enum.hpp",
    }
    for label, path in required.items():
        exists = path is not None and path.is_file()
        report["checks"].append({"name": label, "path": str(path) if path else None, "exists": exists})
        if not exists:
            report["blockers"].append(f"Missing {label}: {path}")
    report["native_build_receipt_exists"] = (CACHE / "package/build_receipt.json").is_file()
    report["sdk_compatibility"] = "Source README requires CANN 9.20; this probe does not establish kernel compiler/runtime compatibility. No device JIT attempted."
    if args.build and not report["blockers"]:
        package_root = CACHE / "package"
        if (package_root / "build_receipt.json").exists():
            raise RuntimeError("Refusing to overwrite a previous native build receipt")
        extension = temp / ("_C" + sysconfig.get_config_var("EXT_SUFFIX"))
        include_dirs = [toolkit / "include", npu_dir / "include", npu_dir / "include/third_party/acl/inc",
                        torch_dir / "include", torch_dir / "include/torch/csrc/api/include",
                        Path(sysconfig.get_paths()["include"]), VENDOR / "deep_gemm/include",
                        VENDOR / "third-party/deep-jit/include", VENDOR / "third-party/magic_enum/include"]
        library_dirs = [toolkit / "lib64", npu_dir / "lib", torch_dir / "lib"]
        command = [compiler, "-std=c++20", "-O3", "-g1", "-fPIC", "-shared", "-Wno-psabi",
                   "-Wno-deprecated-declarations", f"-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}",
                   "-DTORCH_EXTENSION_NAME=_C", "-DTORCH_API_INCLUDE_EXTENSION_H"]
        command += [f"-I{x}" for x in include_dirs]
        command += [str(VENDOR / "csrc/python_api.cpp"), "-o", str(extension)]
        command += [f"-L{x}" for x in library_dirs]
        command += [f"-Wl,-rpath,{x}" for x in library_dirs]
        command += [f"-l{x}" for x in ("torch", "torch_cpu", "torch_python", "c10", "ascendcl", "hccl_fwk", "torch_npu", "opapi", "dl")]
        command += ["-Wl,-Bsymbolic-functions"]
        result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=temp)
        (temp / "compile.stdout").write_text(result.stdout)
        (temp / "compile.stderr").write_text(result.stderr)
        report["checks"].append({"name": "host_extension_build", "command": command, "returncode": result.returncode,
                                 "stdout": str(temp / "compile.stdout"), "stderr": str(temp / "compile.stderr")})
        if result.returncode:
            report["blockers"].append(f"Host extension compilation failed; see {temp / 'compile.stderr'}")
        else:
            target = package_root / "deep_gemm"
            target.mkdir(parents=True, exist_ok=True)
            for src in (VENDOR / "deep_gemm").rglob("*"):
                if src.is_file() and src.suffix in (".py", ".hpp", ".h"):
                    dst = target / src.relative_to(VENDOR / "deep_gemm")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            shutil.copy2(extension, target / extension.name)
            receipt = {
                "schema": "deepgemm-local-build/v1", "machine": platform.machine(),
                "python_cache_tag": sys.implementation.cache_tag, "source_sha256": fingerprint(),
                "command": command, "compiler": report["compiler_version"], "cxx11_abi": args.cxx11_abi,
                "binaries": {"deep_gemm/" + extension.name: sha256(target / extension.name)},
                "device_import_or_execution": False,
            }
            (package_root / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            report["native_build_receipt_exists"] = True
    report["status"] = "deployment_blocked" if report["blockers"] or not report["native_build_receipt_exists"] else "host_built_unverified"
    out = args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "blockers": report["blockers"], "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
