"""Observe original DeepJIT diagnostics without replacing its compiler or kernels.

Import is stdlib-only and does not open a runtime library. Live collection must
run in the externally guarded device process. This is a *load provenance*
receipt, not an ACL launch/loaded-bytes attestation or a numerical result.
One candidate per fresh process is intentional: DeepJIT has a process singleton.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import struct
import sys
import tempfile
import threading
import time


class ProvenanceError(RuntimeError):
    pass


SCHEMA = "deepep-jit-load-provenance/v1"
TEMPLATES = {
    "dispatch": ("ep/dispatch.hpp", "ep/dispatch.hpp"),
    "dispatch_copy_epilogue": ("ep/dispatch.hpp", "ep/dispatch_copy_epilogue.hpp"),
    "combine": ("ep/combine.hpp", "ep/combine.hpp"),
    "combine_reduce_epilogue": ("ep/combine.hpp", "ep/combine_reduce_epilogue.hpp"),
    "engram_fetch": ("engram/engram.hpp", "engram/engram_fetch.hpp"),
    "barrier": ("comm/barrier.hpp", "comm/barrier.hpp"),
}
REQUIRED = {
    "AddFwdOp": {"combine", "combine_reduce_epilogue"},
    "CountNonzeroFwdOp": {"dispatch", "dispatch_copy_epilogue"},
    "MoEPrePermuteFwdOp": {"dispatch", "dispatch_copy_epilogue"},
    "MoEPermuteAlignFwdOp": {"dispatch", "dispatch_copy_epilogue"},
    "PagedKVCacheGatherFwdOp": {"engram_fetch"},
}
ENTRY = re.compile(r"([A-Za-z0-9_]+)\.([0-9a-f]{32})")
_CAPTURE_ACTIVE = False


def require(condition, message):
    if not condition:
        raise ProvenanceError(message)


def file_record(path, *, cache=False):
    """Pin content AND inode identity; reject changes during the read."""
    path = Path(path).absolute()
    if cache:
        require(not any(p.is_symlink() for p in (path, *path.parents)),
                f"symlink in JIT artifact path: {path}")
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(resolved, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        if cache:
            require(before.st_nlink == 1 and before.st_uid == os.getuid(),
                    f"foreign/shared JIT artifact: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        require(all(getattr(before, f) == getattr(after, f) == getattr(resolved.stat(), f)
                    for f in fields), f"file changed while hashing: {path}")
        require(path.resolve(strict=True) == resolved, f"path target changed: {path}")
        return {"path": str(path), "resolved": str(resolved), "sha256": digest.hexdigest(),
                **{f[3:]: getattr(after, f) for f in fields}}
    finally:
        os.close(fd)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def toolkit_paths():
    # Match the pinned Ascend::find_ascend_toolkit order; never import torch/ACL.
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        home = next((p for p in ("/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann")
                     if Path(p).exists()), "")
    require(bool(home), "cannot locate DeepJIT compiler without importing a device runtime")
    root = Path(os.path.abspath(home))
    return {"compiler": root / "bin/bisheng", "linker": root / "bin/ld.lld",
            "adv_include": root / "aarch64-linux/asc/include/adv_api"}


class StdoutCapture:
    """Capture fd 1, including C stdio; redirect_stdout alone misses JIT logs.

    Loading the *current process* with ctypes is only to flush libc. This class
    never opens ACL/vendor DSOs. Its start method is never called at import.
    """

    def __init__(self, path):
        self.path, self.saved_fd, self.stream = Path(path), None, None
        self.flush_c = None

    def flush(self):
        sys.stdout.flush()
        if self.flush_c is not None:
            require(self.flush_c(None) == 0, "libc fflush failed")

    def start(self):
        global _CAPTURE_ACTIVE
        require(not _CAPTURE_ACTIVE, "concurrent stdout provenance capture")
        require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1,
                "start DeepEP provenance in a dedicated single-Python-thread process")
        import ctypes

        self.flush_c = ctypes.CDLL(None).fflush
        self.flush_c.argtypes, self.flush_c.restype = [ctypes.c_void_p], ctypes.c_int
        self.flush()
        self.stream = self.path.open("xb", buffering=0)
        self.saved_fd = os.dup(1)
        try:
            os.dup2(self.stream.fileno(), 1)
        except BaseException:
            os.close(self.saved_fd)
            self.saved_fd = None
            self.stream.close()
            raise
        _CAPTURE_ACTIVE = True

    def read(self, offset):
        self.flush()
        with self.path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
        return data, offset + len(data)

    def stop(self):
        global _CAPTURE_ACTIVE
        if self.saved_fd is None:
            return
        try:
            self.flush()
        finally:
            os.dup2(self.saved_fd, 1)
            os.close(self.saved_fd)
            self.saved_fd = None
            self.stream.close()
            _CAPTURE_ACTIVE = False


def parse_log(text):
    """Start is before ACL load; completion is after load + GetFunction success."""
    pending, complete, commands = None, [], []
    for line in text.splitlines():
        if line.startswith("Running command: "):
            commands.append(line.removeprefix("Running command: "))
        elif line.startswith("Loading Ascend binary: "):
            require(pending is None, "interleaved/unmatched JIT load starts")
            pending = Path(line.removeprefix("Loading Ascend binary: "))
            require(pending.is_absolute() and pending.name == "kernel.o", "invalid JIT load path")
        elif line.startswith("Load time ("):
            match = re.fullmatch(r"Load time \((.*)\): ([0-9]+\.[0-9]+) ms", line)
            require(match is not None and pending is not None, "orphan/malformed JIT load completion")
            require(str(pending.parent) == match[1], "JIT start/completion path mismatch")
            complete.append(str(pending.parent))
            pending = None
    require(pending is None, "JIT load started without successful completion")
    require(len(complete) == len(set(complete)), "duplicate load in one serialized scope")
    return complete, commands


def elf_metadata(path):
    """Read the same section names as the original Kernel::parse_kernel_names.

    This validates structure only; synthetic ELF fixtures remain CPU mocks.
    """
    data = Path(path).read_bytes()
    require(len(data) >= 64 and data[:6] == b"\x7fELF\x02\x01", "invalid Ascend ELF64 little-endian header")
    machine = struct.unpack_from("<H", data, 18)[0]
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, string_index = struct.unpack_from("<HHH", data, 58)
    require(entry_size == 64 and count > 0 and offset + count * 64 <= len(data),
            "invalid/truncated ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * 64) for i in range(count)]
    if string_index == 0xffff:
        string_index = sections[0][6]
    require(string_index < count, "invalid ELF section-name table")
    names_start, names_size = sections[string_index][4:6]
    require(names_start + names_size <= len(data), "truncated ELF section-name table")
    names = data[names_start:names_start + names_size]
    kernels = set()
    for section in sections:
        start = section[0]
        if start >= len(names):
            continue
        end = names.find(b"\x00", start)
        if end < 0:
            continue
        name = names[start:end]
        if name.startswith(b".ascend.meta."):
            kernel = name.removeprefix(b".ascend.meta.").decode("utf-8")
            for suffix in ("_mix_aic", "_mix_aiv"):
                kernel = kernel.removesuffix(suffix)
            kernels.add(kernel)
    require(len(kernels) == 1 and all(kernels), "expected exactly one Ascend kernel metadata name")
    return {"elf_machine": machine, "kernel_name": next(iter(kernels))}


def generated_template_parameters(host, tag, source):
    """Match the pinned C++ format literal; permit only native numeric arguments."""
    pattern = r'jit->compile\("' + re.escape(tag) + r'", std::format\(R"\((.*?)\)"'
    matches = re.findall(pattern, Path(host).read_text(), re.DOTALL)
    require(len(matches) == 1, "missing/ambiguous original JIT source format literal")
    literal, regex, index = matches[0], [], 0
    while index < len(literal):
        pair = literal[index:index + 2]
        if pair in ("{{", "}}"):
            regex.append(re.escape(pair[0]))
            index += 2
        elif pair == "{}":
            regex.append(r"(-?[0-9]+(?:, -?[0-9]+)*|true|false)")
            index += 2
        else:
            require(literal[index] not in "{}", "unsupported native format expression")
            regex.append(re.escape(literal[index]))
            index += 1
    match = re.fullmatch("".join(regex), source)
    require(match is not None, "generated source differs from the original host template")
    return list(match.groups())


def _flags(options):
    level, arch = options.get("optimize_level"), options.get("arch")
    require(isinstance(level, str) and level and type(arch) is int and arch > 0,
            "invalid compiler options")
    bisheng = ["-O" + level, "--npu-arch=dav-" + str(arch)]
    if options.get("debug_info"):
        bisheng += ["-g", "-gdwarf-5", "-gembed-source", "-gline-tables-only"]
    for key in ("bisheng_flags", "extra_bisheng_flags"):
        value = options.get(key) or []
        require(isinstance(value, list) and all(isinstance(x, str) for x in value), "invalid flags")
        bisheng += value
    linker = []
    for key in ("linker_flags", "extra_linker_flags"):
        value = options.get(key) or []
        require(isinstance(value, list) and all(isinstance(x, str) for x in value), "invalid flags")
        linker += value
    require(options.get("dump_asm") is False, "provenance requires EP_JIT_DUMP_ASM=0")
    return bisheng, linker


class JITProvenance:
    """Fail closed on unobserved/polluted caches and persist immutable receipts.

    Source fingerprints are sampled before initialization and at every scope.
    New kernel bytes can first be hashed *after* the call returns; see limits.
    Constructor only inspects files and sets observational environment options.
    """

    def __init__(self, root, workspace, vendor, build, identity, sources, *, tools=None):
        self.root, self.workspace, self.vendor, self.build = map(
            lambda p: Path(p).absolute(), (root, workspace, vendor, build))
        require(identity.get("op") in REQUIRED and isinstance(identity.get("case_id"), str)
                and identity["case_id"], "missing supported op/case identity")
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.poisoned, self.finished, self.in_run = None, False, False
        self.runs, self.offset, self.scopes = 0, 0, []
        self.entries, self.run_entries = {}, set()
        self.source_hashes = sources  # SAME mapping exported by Candidate.source_sha256.
        self.expected_hashes, self.sealed = {}, {}
        self.tools = tools or toolkit_paths()
        require(self.root.resolve() == self.root and not self.root.is_symlink(), "noncanonical JIT root")
        self.root_stat = self.root.stat()
        require(self.root_stat.st_uid == os.getuid() and stat.S_IMODE(self.root_stat.st_mode) == 0o700,
                "JIT root must be owned mode 0700")
        require(not list(self.root.iterdir()), "JIT root is not empty before this case")
        require(os.environ.get("EP_JIT_CACHE_DIR") == str(self.root), "foreign JIT cache environment")
        self.environment_before = {k: os.environ.get(k) for k in (
            "EP_JIT_DEBUG", "EP_JIT_PRINT_COMPILER_COMMAND", "EP_JIT_PRINT_LOAD_TIME", "EP_JIT_DUMP_ASM")}
        for key, value in {"EP_JIT_DEBUG": "1", "EP_JIT_PRINT_COMPILER_COMMAND": "1",
                           "EP_JIT_PRINT_LOAD_TIME": "1", "EP_JIT_DUMP_ASM": "0"}.items():
            os.environ[key] = value
        # Preserve only relevant toolchain/JIT controls, never arbitrary credentials.
        self.environment = {k: v for k, v in os.environ.items()
                            if k.startswith(("EP_JIT_", "DJ_JIT_")) or k in (
                                "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_LAUNCH_BLOCKING")}
        self.pinned = {}
        for name, expected in list(sources.items()):
            path = Path(name) if Path(name).is_absolute() else self.workspace / name
            record = file_record(path)
            require(record["sha256"] == expected, f"source changed before provenance: {path}")
            self.pinned[str(path)] = record
        # Pin staged source templates separately from the original source tree.
        self.staged_files = sorted(str(p) for p in (self.build / "deep_ep").rglob("*")
                                   if p.is_file() and p.suffix in (".py", ".hpp"))
        require(bool(self.staged_files), "no staged DeepEP source templates")
        for name in self.staged_files:
            self.pinned[name] = file_record(name)
        for key in ("compiler", "linker"):
            path = self.tools[key]
            require(os.access(path, os.X_OK), f"compiler/linker is not executable: {path}")
            self.pinned[str(path)] = file_record(path)
        require(Path(self.tools["adv_include"]).is_dir(), "missing Ascend adv_api include directory")
        self.evidence = Path(tempfile.mkdtemp(prefix="jit-evidence-", dir=self.root.parent))
        self.capture = StdoutCapture(self.evidence / "stdout.raw.log")
        self._publish_hashes(self.pinned.values())
        _write_new(self.evidence / "start.json", {
            "schema": SCHEMA, "case": self.identity, "pid": self.pid, "cache_root": str(self.root),
            "vendor_root": str(self.vendor), "build_root": str(self.build),
            "initial_cache_empty": True, "environment": self.environment,
            "environment_before_observation": self.environment_before,
            "source_files_before": self.pinned, "toolkit": {k: str(v) for k, v in self.tools.items()},
            "monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
        })
        self._publish_hashes([file_record(self.evidence / "start.json")])

    def _publish_hashes(self, records):
        for record in records:
            path = Path(record["path"])
            key = str(path.relative_to(self.workspace)) if path.is_relative_to(self.workspace) else str(path)
            require(key not in self.source_hashes or self.source_hashes[key] == record["sha256"],
                    f"Candidate source digest conflict: {key}")
            self.source_hashes[key] = record["sha256"]
            self.expected_hashes[key] = record["sha256"]
            if path.is_relative_to(self.evidence):
                self.sealed[str(path)] = record

    def start(self):
        self.capture.start()

    def _stable(self):
        require(os.getpid() == self.pid and threading.get_ident() == self.thread,
                "JIT provenance owner process/thread changed")
        require(threading.active_count() == 1, "concurrent Python threads invalidate case-scoped stdout attribution")
        require(not self.finished and self.poisoned is None, "closed/rejected JIT provenance session")
        require(self.root.resolve() == self.root and self.root.stat().st_ino == self.root_stat.st_ino
                and self.root.stat().st_dev == self.root_stat.st_dev
                and stat.S_IMODE(self.root.stat().st_mode) == 0o700, "JIT root replaced/permissions changed")
        current = {k: v for k, v in os.environ.items()
                   if k.startswith(("EP_JIT_", "DJ_JIT_")) or k in (
                       "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_LAUNCH_BLOCKING")}
        require(current == self.environment and current.get("EP_JIT_CACHE_DIR") == str(self.root),
                "JIT/compiler environment changed during case")
        require(sorted(str(p) for p in (self.build / "deep_ep").rglob("*")
                       if p.is_file() and p.suffix in (".py", ".hpp")) == self.staged_files,
                "staged template file set changed")
        for name, before in self.pinned.items():
            require(file_record(name) == before, f"source/compiler replaced or changed: {name}")
        for name, before in self.sealed.items():
            require(file_record(name) == before, f"sealed provenance evidence changed: {name}")
        require(all(self.source_hashes.get(k) == v for k, v in self.expected_hashes.items()),
                "Candidate source_sha256 mapping was changed or truncated")

    def _cache(self):
        entries = {}
        require(set(p.name for p in self.root.iterdir()) <= {"cache", "tmp"}, "unexpected JIT root files")
        for directory in self.root.rglob("*"):
            require(not directory.is_symlink(), f"symlink in JIT cache: {directory}")
        tmp = self.root / "tmp"
        require(not tmp.exists() or (tmp.is_dir() and not list(tmp.iterdir())),
                "uncommitted/unfinished JIT temporary")
        cache = self.root / "cache"
        if not cache.exists():
            return entries
        require(cache.is_dir(), "cache is not a directory")
        for entry in sorted(cache.iterdir()):
            match = ENTRY.fullmatch(entry.name)
            require(entry.is_dir() and match is not None and match[1] in TEMPLATES, "unknown cache entry")
            require(set(p.name for p in entry.iterdir()) == {"kernel.o", "kernel.asc", "meta.json", ".committed"},
                    f"incomplete/unexpected cache artifacts: {entry}")
            records = {p.name: file_record(p, cache=True) for p in sorted(entry.iterdir())}
            require(records[".committed"]["size"] == 0, "nonempty cache commit marker")
            require(records["kernel.o"]["size"] > 0, "empty JIT binary")
            entries[str(entry)] = {"tag": match[1], "cache_key": match[2], "files": records}
        return entries

    def _metadata(self, path, record, commands, lines):
        path = Path(path)
        meta = read_json(path / "meta.json")
        require(meta.get("config") == {"python_library_root": str(self.build / "deep_ep"),
                    "extra_signature": "", "include_dirs": [str(self.build / "deep_ep/include")],
                    "include_prefixes": ["deep_ep/"]}, "foreign JIT template/config root")
        compiler = meta.get("compiler_info", {})
        require(compiler.get("path") == str(self.tools["compiler"])
                and compiler.get("linker_path") == str(self.tools["linker"])
                and isinstance(compiler.get("version"), str) and compiler["version"]
                and isinstance(compiler.get("linker_version"), str) and compiler["linker_version"],
                "missing/foreign compiler metadata")
        bisheng, linker = _flags(meta.get("compiler_options", {}))
        # Published metadata retains the pre-rename temporary paths. Associate
        # it through identical logged compiler/linker commands, not a basename.
        command = meta.get("command", "")
        parts = command.split(" && ")
        require(len(parts) == 2, "missing exact compiler/linker command pair")
        tokens = [shlex.split(p) for p in parts]
        require(all(" ".join(t) == p for t, p in zip(tokens, parts)), "ambiguous JIT command quoting")
        require(len(tokens[0]) >= 4 and tokens[0][-2] == "-o", "invalid compile command output")
        temporary = Path(tokens[0][-3]).parent
        require(temporary.parent == self.root / "tmp" and temporary.name not in ("", ".", ".."),
                "compile command was not in this private JIT root")
        expected_compile = [str(self.tools["compiler"]), *bisheng,
                            "-I" + str(self.build / "deep_ep/include"),
                            "-I" + str(self.tools["adv_include"]),
                            str(temporary / "kernel.asc"), "-o", str(temporary / "kernel.rel.o")]
        expected_link = [str(self.tools["linker"]), *linker, str(temporary / "kernel.rel.o"),
                         "-o", str(temporary / "kernel.o")]
        require(tokens == [expected_compile, expected_link], "metadata command/flags mismatch")
        observed = ["cd " + str(temporary) + " && " + p for p in parts]
        require(all(commands.count(p) == 1 for p in observed), "missing/duplicate observed compilation commands")
        require(commands.index(observed[0]) < commands.index(observed[1]), "link logged before compile")
        require(lines.index("Running command: " + observed[1]) <
                lines.index("Loading Ascend binary: " + str(path / "kernel.o")),
                "load preceded successful compilation command sequence")
        host, header = TEMPLATES[record["tag"]]
        host = self.vendor / "csrc/kernels" / host
        header = self.build / "deep_ep/include/deep_ep/impls" / header
        require(str(host) in self.pinned and str(header) in self.pinned, "template source was not pinned")
        source = (path / "kernel.asc").read_text()
        includes = re.findall(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", source, re.MULTILINE)
        expected_header = "deep_ep/impls/" + TEMPLATES[record["tag"]][1]
        require(includes == [expected_header], "generated kernel does not reference the expected template")
        parameters = generated_template_parameters(host, record["tag"], source)
        return {"metadata": meta, "compile_argv": expected_compile, "link_argv": expected_link,
                "compile_working_directory": str(temporary), "observed_commands": observed,
                "host_template": self.pinned[str(host)], "device_template": self.pinned[str(header)],
                "compiler_file": self.pinned[str(self.tools["compiler"])],
                "linker_file": self.pinned[str(self.tools["linker"])],
                "template_arguments_as_emitted": parameters,
                "elf": elf_metadata(path / "kernel.o"),
                "first_binary_hash_observation": "after_successful_scoped_call_and_synchronization"}

    def checkpoint(self, phase, *, require_run=False):
        """Seal the preceding scope. Caller must synchronize before this call."""
        try:
            require(phase in ("initialization", "factory_preparation", "before_run", "candidate_run", "cleanup")
                    and require_run == (phase == "candidate_run"), "invalid provenance phase/run marker")
            self._stable()
            data, end = self.capture.read(self.offset)
            decoded = data.decode("utf-8", errors="strict")
            loads, commands = parse_log(decoded)
            current = self._cache()
            for name, previous in self.entries.items():
                require(name in current and current[name] == previous["snapshot"],
                        f"previous JIT object/marker/source/metadata replaced or removed: {name}")
            new = set(current) - set(self.entries)
            require(set(loads) == new, "cache objects and successful loads are not one-to-one in this case scope")
            require(phase != "before_run" or not new,
                    "unscoped JIT compilation/load occurred between candidate invocations")
            associations = {name: self._metadata(name, current[name], commands, decoded.splitlines())
                            for name in sorted(new)}
            expected_commands = [c for value in associations.values() for c in value["observed_commands"]]
            require(sorted(commands) == sorted(expected_commands), "unattributed compiler command in case scope")
            require(self._cache() == current, "JIT artifacts changed while validating metadata")
            if require_run and self.runs == 0:
                tags = {current[name]["tag"] for name in new}
                require(REQUIRED[self.identity["op"]] <= tags,
                        "first candidate run lacks this case's required successful JIT loads")
            if require_run:
                self.run_entries.update(new)
                self.runs += 1
            index = len(self.scopes)
            log = self.evidence / f"scope-{index:04d}.log"
            with log.open("xb") as stream:
                stream.write(data)
            scope = {"schema": SCHEMA, "case": self.identity, "phase": phase,
                     "index": index, "pid": self.pid, "run_index": self.runs if require_run else None,
                     "stdout_byte_range": [self.offset, end], "stdout": file_record(log),
                     "successful_loads": loads, "new_entries": {name: current[name] for name in sorted(new)},
                     "associations": associations, "known_entries_stable": sorted(self.entries),
                     "run_returned_and_synchronized": require_run, "monotonic_ns": time.monotonic_ns(),
                     "direct_acl_launch_attestation": False}
            receipt = self.evidence / f"scope-{index:04d}.json"
            _write_new(receipt, scope)
            self.scopes.append(str(receipt))
            for name in new:
                self.entries[name] = {"snapshot": current[name], "first_scope": index}
                self._publish_hashes(current[name]["files"].values())
            self._publish_hashes([file_record(log), file_record(receipt)])
            self.offset = end
            return scope
        except BaseException as exc:
            self.poisoned = f"{type(exc).__name__}: {exc}"
            raise

    def wrap(self, run, synchronize):
        def observed(*args, **kwargs):
            require(not self.in_run, "reentrant/concurrent DeepEP candidate invocation")
            self.in_run = True
            try:
                # Attribute no activity between invocations to the next run.
                self.checkpoint("before_run")
                result = run(*args, **kwargs)
                synchronize()
                self.checkpoint("candidate_run", require_run=True)
                return result
            except BaseException as exc:
                self.poisoned = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.in_run = False
        return observed

    def finish(self, error=None):
        if self.finished:
            return
        failure = error or self.poisoned
        try:
            if failure is None:
                self.checkpoint("cleanup")
                require(self.runs > 0 and self.run_entries, "no successful case run with observed JIT loads")
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            self.capture.stop()
            self.finished = True
        receipt = self.evidence / "final.json"
        _write_new(receipt, {
            "schema": SCHEMA, "case": self.identity, "pid": self.pid,
            "status": "rejected" if failure else "load_provenance_collected",
            "error": str(failure) if failure else None, "cache_root": str(self.root),
            "runs": self.runs, "scopes": self.scopes, "entries": self.entries,
            "candidate_run_entries": sorted(self.run_entries), "stdout": file_record(self.capture.path),
            "native_pass": False, "direct_acl_launch_attestation": False,
            "loaded_byte_identity_attested": False, "toolchain_dependencies_complete": False,
            "limits": [
                "Official logs identify successful BinaryLoadFromFile/GetFunction, not launch handles.",
                "New objects are first hashed after scoped execution; no atomic pre-load byte hash hook exists.",
                "Warm calls inherit this case's observed load set; no per-launch attribution is claimed.",
                "No defense against same-uid malicious log forgery or transient replace-and-restore attacks.",
                "Vendor/staged templates and compiler/linker are pinned; transitive CANN/sysroot dependencies are not complete.",
                "Numerical/input-preservation verification is the independent shared checker's responsibility.",
                "Provenance hashing/log capture overhead makes this run unsuitable for latency reporting.",
            ],
        })
        self._publish_hashes([file_record(self.capture.path), file_record(receipt)])
        if failure:
            raise ProvenanceError(str(failure))
