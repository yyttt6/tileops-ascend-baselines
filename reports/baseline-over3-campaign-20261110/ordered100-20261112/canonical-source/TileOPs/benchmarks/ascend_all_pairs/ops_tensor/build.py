"""CPU-only compilation of unchanged vendor examples; never loads the result.

All outputs and compiler temporary files stay under this worker's cache. Importing
this module uses only the Python standard library and does not touch a device.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
VENDOR = WORKSPACE / "ascend_baselines/ops-tensor"
CACHE = WORKSPACE / ".cache/all-pairs/ops_tensor"
REG = "include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute"
SOURCES = {
    "add": f"{REG}/add/add.asc",
    "trunc": f"{REG}/trunc/trunc.asc",
    "reduction": f"{REG}/reduction_compute/reduction_compute.asc",
    "gemm": "examples/mat_mul/mat_mul_basic/mat_mul_basic.cpp",
}


class DeploymentBlocked(RuntimeError):
    """An implemented mathematical route lacks a compatible deployment."""


def fingerprint(paths):
    result = {}
    for path in sorted(set(Path(p).resolve() for p in paths)):
        key = str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)
        result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_files(kind):
    roots = [VENDOR / "include/tensor_api/include"]
    if kind == "gemm":
        roots.append(VENDOR / "include/blaze")
    files = [HERE / "native" / f"{kind}.cpp", HERE / "build.py", VENDOR / SOURCES[kind]]
    for root in roots:
        files.extend(p for p in root.rglob("*") if p.suffix in (".h", ".hpp"))
    files.extend((VENDOR / "examples/common").glob("*.h"))
    files.extend((VENDOR / SOURCES[kind]).parent.glob("*.h"))
    return files


def build_spec(kind):
    if kind not in SOURCES:
        raise ValueError(kind)
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")).resolve()
    compiler = shutil.which("bisheng") or str(sdk / "bin/bisheng")
    if not Path(compiler).is_file():
        raise DeploymentBlocked(f"Missing compiler: {compiler}")
    arch = platform.machine()
    if arch not in ("aarch64", "x86_64"):
        raise DeploymentBlocked(f"Unsupported host ABI {arch}")
    includes = [VENDOR, VENDOR / "include", VENDOR / "include/blaze",
                VENDOR / "include/tensor_api/include", VENDOR / "include/tensor_api",
                (VENDOR / SOURCES[kind]).parent, VENDOR / "examples", VENDOR / "examples/common",
                sdk / "include", sdk / "include/op_common", sdk / "pkg_inc/op_common",
                sdk / "pkg_inc/base", sdk / "pkg_inc", sdk / "asc/include",
                sdk / f"{arch}-linux/asc", sdk / f"{arch}-linux/asc/include",
                sdk / f"{arch}-linux/asc/include/tiling", sdk / "asc/impl/c_api",
                sdk / "asc/impl/utils", sdk / "compiler/ascendc/include/highlevel_api"]
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared", "-fPIC"]
    if kind != "gemm":
        flags += ["-DCANN_ASC_USE_EXPERIMENTAL", "-DSCENARIO_NUM=1"]
    libs = ["ascendcl", "ascendc_runtime", "runtime"]
    if kind == "gemm":
        libs += ["tiling_api", "platform", "mmpa", "c_sec", "dl"]
    sdk_evidence = [Path(compiler).resolve()]
    for rel in ("version.info", f"{arch}-linux/ascend_toolkit_install.info",
                f"{arch}-linux/asc/impl/basic_api/kernel_tensor.h"):
        if (sdk / rel).is_file():
            sdk_evidence.append(sdk / rel)
    hashes = fingerprint(source_files(kind) + sdk_evidence)
    identity = {"kind": kind, "host_arch": arch, "sdk": str(sdk), "compiler": compiler,
                "flags": flags, "includes": list(map(str, includes)), "libraries": libs,
                "source_sha256": hashes}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = CACHE / f"{kind}-{key}"
    output = directory / "libopstensor.so"
    command = [compiler, *flags, str(HERE / "native" / f"{kind}.cpp"), "-o", str(output)]
    for include in includes:
        command += ["-I", str(include)]
    command += ["-iquote" + str(VENDOR / "include/tensor_api/include"),
                "-iquote" + str(VENDOR / "include/tensor_api"),
                "-L", str(sdk / "lib64"), "-L", str(sdk / f"{arch}-linux/lib64")]
    command += ["-l" + name for name in libs]
    return identity | {"cache_directory": str(directory), "binary": str(output), "command": command}


def checked_elf_machine(binary, host_arch):
    header = Path(binary).read_bytes()[:20]
    expected = {"aarch64": 183, "x86_64": 62}.get(host_arch)
    if (len(header) != 20 or header[:4] != b"\x7fELF" or
            header[4:6] != bytes((2, 1)) or expected is None):
        raise DeploymentBlocked(f"Invalid ELF64 little-endian artifact: {binary}")
    machine = struct.unpack("<H", header[18:20])[0]
    if machine != expected:
        raise DeploymentBlocked(f"Wrong host ELF machine: {machine}, expected {expected}")
    return machine


def compile_native(kind, timeout=600):
    """Compile/link only. No ACL import, dlopen, subprocess execution of binary, or NPU probe."""
    spec = build_spec(kind)
    directory = Path(spec["cache_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    binary = Path(spec["binary"])
    receipt = directory / "build.json"
    if receipt.exists() and binary.exists():
        old = json.loads(receipt.read_text())
        if old.get("status") == "compiled":
            if (any(old.get(key) != value for key, value in spec.items()) or
                    old.get("binary_sha256") != fingerprint([binary])):
                raise DeploymentBlocked(f"Cached build identity/hash mismatch: {receipt}")
            checked_elf_machine(binary, spec["host_arch"])
            return binary, old["source_sha256"] | old["binary_sha256"] | fingerprint([receipt])
    env = dict(os.environ)
    env["TMPDIR"] = str(directory)
    env["TMP"] = str(directory)
    env["TEMP"] = str(directory)
    # Compiler subprocesses only; intentionally never load the generated library.
    try:
        result = subprocess.run(spec["command"], cwd=directory, env=env,
                                capture_output=True, text=True, timeout=timeout)
        (directory / "build.log").write_text(result.stdout + result.stderr)
        spec.update(returncode=result.returncode, status="compiled" if result.returncode == 0 else "deployment_blocked")
    except subprocess.TimeoutExpired as exc:
        log = (exc.stdout or b"") + (exc.stderr or b"")
        (directory / "build.log").write_bytes(log if isinstance(log, bytes) else log.encode())
        spec.update(returncode=None, status="deployment_blocked", reason=f"Compiler timeout after {timeout}s")
    if spec["status"] == "compiled":
        try:
            machine = checked_elf_machine(binary, spec["host_arch"])
        except DeploymentBlocked as exc:
            spec.update(status="deployment_blocked", reason=str(exc))
        else:
            spec.update(binary_sha256=fingerprint([binary]), elf_machine=machine)
    spec.update(npu_initialized=False, binary_loaded=False, device_validated=False)
    receipt.write_text(json.dumps(spec, indent=2) + "\n")
    if spec["status"] != "compiled":
        raise DeploymentBlocked(f"{kind} compilation blocked; see {directory / 'build.log'}")
    return binary, spec["source_sha256"] | spec["binary_sha256"] | fingerprint([receipt])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=tuple(SOURCES))
    args = parser.parse_args()
    binary, _ = compile_native(args.kind)
    print(binary)
