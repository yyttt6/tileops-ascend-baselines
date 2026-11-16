"""CPU-only compatibility build for original AtlasA2 code on the target3510 SDK.

This never loads a binary or changes the vendor architecture tags/schedule.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from .build import HERE, VENDOR, CACHE, hashes, toolchain


def sources():
    files = [Path(__file__), HERE / "native/mla.cpp", HERE / "build.py"]
    for root in (VENDOR / "include", VENDOR / "tests/optest/include", VENDOR / "tests/optest/kernels/19_mla"):
        files.extend(p for p in root.rglob("*") if p.suffix in (".h", ".hpp", ".cpp"))
    return files


def build():
    sdk, compiler = toolchain()
    shas = hashes(sources())
    compiler_sha = hashlib.sha256(compiler.read_bytes()).hexdigest()
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared", "-fPIC",
             "-DCATLASS_ARCH=3510", "-DTILING_KEY_VAR", "-Wno-macro-redefined",
             "-mllvm", "-cce-aicore-record-overflow=true", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false"]
    fingerprint = hashlib.sha256(json.dumps([shas, str(compiler), compiler_sha, str(sdk), flags], sort_keys=True).encode()).hexdigest()
    folder = CACHE / "legacy" / ("mla-" + fingerprint[:20])
    folder.mkdir(parents=True, exist_ok=True)
    binary, report_path = folder / "libcatlass_mla.so", folder / "build.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["returncode"]:
            raise RuntimeError(f"CATLASS AtlasA2-to3510 build blocked; recorded original compiler error: {folder / 'build.log'}")
        if binary.exists() and hashlib.sha256(binary.read_bytes()).hexdigest() == report["binary_sha256"]:
            return binary, shas | hashes([binary]), report
    command = [str(compiler), *flags, str(HERE / "native/mla.cpp"), "-o", str(binary)]
    for path in (VENDOR / "include", VENDOR / "tests/optest/include", VENDOR / "tests/optest/kernels",
                 VENDOR / "tests/optest/kernels/19_mla", VENDOR / "tests/optest/kernels/19_mla/tiling", sdk / "include"):
        command += ["-I", str(path)]
    command += ["-L", str(sdk / "lib64"), "-ltiling_api", "-lplatform", "-lascendcl", "-ldl"]
    temporary = folder / "tmp"
    temporary.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary), ASCEND_HOME_PATH=str(sdk))
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=folder, env=env)
        output, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        output, code = str(exc), 124
    (folder / "build.log").write_text(output)
    report = {"kind": "mla", "command": command, "returncode": code, "machine": platform.machine(), "sdk": str(sdk),
              "compiler": str(compiler), "compiler_sha256": compiler_sha, "source_sha256": shas,
              "log": str(folder / "build.log"), "device_loaded": False, "device_run": False}
    if not code:
        report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if code:
        raise RuntimeError(f"CATLASS AtlasA2-to3510 build blocked ({code}): {folder / 'build.log'}")
    return binary, shas | hashes([binary]), report
