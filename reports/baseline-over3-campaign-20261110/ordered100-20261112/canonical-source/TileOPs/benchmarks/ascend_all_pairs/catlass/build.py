"""Isolated CATLASS compilation. This module never imports/loads a device runtime.

`python build.py --all` only runs the CANN compiler and records its outputs.
Loading the resulting library is a separate, future device-authorized factory action.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
VENDOR = WORKSPACE / "ascend_baselines/catlass"
CACHE = WORKSPACE / ".cache/all-pairs/catlass"
KINDS = ("gemm", "gemm_f16", "gemm_f32", "gemm_f16_f32", "gemm_bf16_f32", "gemm_bias", "attention_cases", "fp8_f16", "fp8_bf16", "int8")


def hashes(paths):
    return {str(p.relative_to(WORKSPACE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def sources(kind):
    paths = [HERE / "native" / f"{kind}.cpp", Path(__file__)]
    roots = [VENDOR / "include", VENDOR / "examples/common"]
    if kind == "attention_cases":
        roots += [VENDOR / "examples/49_ascend950_flash_attention_infer"]
    elif kind == "int8":
        roots += [VENDOR / "tests/optest/include", VENDOR / "tests/optest/kernels/common",
                  VENDOR / "tests/optest/kernels/57_ascend950_matmul_full_dequant"]
    elif kind.startswith("fp8_"):
        roots += [VENDOR / "tests/optest/include", VENDOR / "tests/optest/kernels/common",
                  VENDOR / "tests/optest/kernels/51_ascend950_quant_matmul_per_group_per_block_tla"]
    else:
        roots += [VENDOR / "tests/optest/include", VENDOR / "tests/optest/kernels/common",
                  VENDOR / "tests/optest/kernels/43_ascend950_basic_matmul"]
        paths += [VENDOR / "examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp"]
    for root in roots:
        paths.extend(p for p in root.rglob("*") if p.suffix in (".hpp", ".h", ".cpp"))
    return paths


def toolchain():
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")).resolve()
    compiler = shutil.which("bisheng")
    if not compiler and (sdk / "bin/bisheng").exists():
        compiler = str(sdk / "bin/bisheng")
    if not compiler:
        raise RuntimeError(f"CATLASS deployment missing compiler: {sdk / 'bin/bisheng'}")
    return sdk, Path(compiler).resolve()


def build(kind):
    if kind not in KINDS:
        raise ValueError(kind)
    sdk, compiler = toolchain()
    source_sha256 = hashes(sources(kind))
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared", "-fPIC",
             "-DCATLASS_ARCH=3510", "-DTILING_KEY_VAR", "-Wno-macro-redefined"]
    fingerprint = [source_sha256, str(compiler), hashlib.sha256(compiler.read_bytes()).hexdigest(),
                   str(sdk), platform.machine(), flags]
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    folder = CACHE / "native" / f"{kind}-{digest[:20]}"
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "libcatlass_all_pairs.so"
    receipt_path = folder / "build.json"
    if binary.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("binary_sha256") == hashlib.sha256(binary.read_bytes()).hexdigest():
            return binary, source_sha256 | hashes([binary]), receipt
    temporary = folder / "libcatlass_all_pairs.building.so"
    includes = [VENDOR / "include", VENDOR / "tests/optest/include", VENDOR / "tests/optest/kernels",
                VENDOR / "examples/common", VENDOR / "examples/49_ascend950_flash_attention_infer",
                sdk / "include", sdk / "include/aclnn"]
    command = [str(compiler), *flags, str(HERE / "native" / f"{kind}.cpp"), "-o", str(temporary)]
    for include in includes:
        command += ["-I", str(include)]
    command += ["-L", str(sdk / "lib64"), "-ltiling_api", "-lplatform", "-lascendcl", "-ldl"]
    env = os.environ.copy()
    temp = folder / "tmp"
    temp.mkdir(exist_ok=True)
    env.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp), ASCEND_HOME_PATH=str(sdk))
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=folder, env=env)
        output, returncode = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        output = str(exc)
        returncode = 124
    (folder / "build.log").write_text(output)
    receipt = {"kind": kind, "command": command, "returncode": returncode, "machine": platform.machine(),
               "sdk": str(sdk), "compiler": str(compiler), "compiler_sha256": fingerprint[2],
               "log": str(folder / "build.log"), "source_sha256": source_sha256,
               "device_loaded": False, "device_run": False}
    if returncode == 0:
        temporary.replace(binary)
        receipt["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if returncode:
        raise RuntimeError(f"CATLASS {kind} compile failed ({returncode}): {folder / 'build.log'}")
    return binary, source_sha256 | hashes([binary]), receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()
    failed = False
    for kind in KINDS if args.all else (args.kind or "gemm_f16",):
        try:
            path, _, _ = build(kind)
            print(f"{kind}: compiled (not loaded) {path}", flush=True)
        except RuntimeError as error:
            failed = True
            print(str(error), flush=True)
    raise SystemExit(1 if failed else 0)
