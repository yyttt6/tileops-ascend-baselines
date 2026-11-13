"""Experimental, explicit lookahead for the AscendC backend; no device work in workers."""

import concurrent.futures
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time


class DeferredLaunch(RuntimeError):
    """Graph is ready; stop before launching its first deferred kernel."""


def compile_artifacts(request):
    # Source generation, device queries, loading and autotuning stay in the parent.
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    import torch_npu  # noqa: F401
    from torch_npu._inductor.ascendc.compiler._compiler import compile_ascendc

    started = time.monotonic_ns()
    kernel = compile_ascendc(
        request["artifacts"],
        request["lib_dir"],
        None,
        soc_version=request["soc_version"],
        compile_flags=request["compile_flags"],
    )
    descriptor = {"lib_wrapper": kernel.lib_wrapper}
    if hasattr(kernel, "kernel_infos"):
        descriptor["kernel_infos"] = [vars(info) for info in kernel.kernel_infos]
    else:
        descriptor.update(lib_kernel=kernel.lib_kernel, kernel_key=kernel.kernel_key)
    paths = [descriptor["lib_wrapper"]]
    paths += (
        [x["kernel"] for x in descriptor["kernel_infos"]]
        if "kernel_infos" in descriptor
        else [descriptor["lib_kernel"]]
    )
    descriptor.update(
        worker_pid=os.getpid(),
        start_mono_ns=started,
        end_mono_ns=time.monotonic_ns(),
        artifact_hashes={
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
        },
    )
    Path(request["receipt"]).write_text(json.dumps(descriptor, indent=2) + "\n")
    return descriptor


class DeferredKernel:
    def __init__(self, stage, future):
        self.stage, self.future = stage, future
        self.kernel = None
        self.descriptor = None
        self.calls = 0

    def __call__(self, *args):
        if not self.stage["enabled"]:
            raise DeferredLaunch(self.stage["name"])
        if self.kernel is None:
            from torch_npu._inductor.ascendc.compiler._kernel import NpuInductorKernel
            from torch_npu._inductor.ascendc.compiler._autotune import (
                NpuInductorAutoTuneKernel,
                KernelInfo,
            )

            d = self.future.result(timeout=600)
            for name, digest in d["artifact_hashes"].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError("compiled artifact changed before consumption")
            self.descriptor = d
            if "kernel_infos" in d:
                self.kernel = NpuInductorAutoTuneKernel(
                    d["lib_wrapper"], [KernelInfo(**x) for x in d["kernel_infos"]]
                )
            else:
                self.kernel = NpuInductorKernel(
                    d["lib_wrapper"], d["lib_kernel"], d["kernel_key"]
                )
        value = self.kernel(*args)
        self.calls += 1
        self.stage["last_consumption"] = {
            "pid": os.getpid(),
            "mono_ns": time.monotonic_ns(),
        }
        return value


class StagedBackend:
    def __init__(self, output, workers=2):
        from torch_npu._inductor.ascendc import compiler
        import torch

        self.compiler = compiler
        self.original = compiler.async_compile
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.soc = torch.npu.get_device_properties().name
        self.flags = compiler.get_compile_flags()
        self.pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.active = None
        self.stages = []
        self.stop = threading.Event()
        self.observations = []
        self.unsafe = threading.Event()
        self.observation_errors = []
        self.observation_incomplete = threading.Event()
        self.proc_races = 0
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()
        compiler.async_compile = self._compile

    def _read_error(self, error, pid, operation):
        # Process exit / FD close between enumeration and read is normal.
        if isinstance(error, OSError) and error.errno in (errno.ENOENT, errno.ESRCH):
            self.proc_races += 1
            return
        self.observation_errors.append(
            {
                "pid": pid,
                "operation": operation,
                "error": repr(error),
                "mono_ns": time.monotonic_ns(),
            }
        )
        self.observation_incomplete.set()

    def _sample_workers(self, seen):
        processes = getattr(self.pool, "_processes", None) or {}
        todo = list(processes)
        visited = set()
        while todo:
            pid = todo.pop()
            if pid in visited:
                continue
            visited.add(pid)
            try:
                todo.extend(
                    map(
                        int,
                        Path(f"/proc/{pid}/task/{pid}/children").read_text().split(),
                    )
                )
                for fd in Path(f"/proc/{pid}/fd").iterdir():
                    try:
                        target = os.readlink(fd)
                    except OSError as error:
                        self._read_error(error, pid, str(fd))
                        continue
                    if (
                        target.startswith(("/dev/davinci", "/dev/devmm"))
                        and (pid, target) not in seen
                    ):
                        seen.add((pid, target))
                        self.observations.append(
                            {
                                "pid": pid,
                                "target": target,
                                "mono_ns": time.monotonic_ns(),
                            }
                        )
                        if target != "/dev/davinci_manager":
                            self.unsafe.set()
            except (OSError, ValueError) as error:
                self._read_error(error, pid, "process tree / FD enumeration")

    def _monitor(self):
        seen = set()
        try:
            while not self.stop.wait(0.05):
                self._sample_workers(seen)
                if self.observation_incomplete.is_set():
                    return
        except Exception as error:
            self._read_error(error, None, "monitor thread")

    def check(self):
        if self.observation_incomplete.is_set():
            raise RuntimeError("worker device observation incomplete; pipeline invalid")
        if self.unsafe.is_set():
            raise RuntimeError(
                "backend worker accessed compute device; pipeline invalid"
            )

    def _compile(self, executor, artifacts):
        if self.active is None:
            raise RuntimeError("unexpected new compilation outside graph preparation")
        stage = self.active
        number = len(stage["kernels"])
        directory = self.output / stage["name"] / f"kernel{number}"
        directory.mkdir(parents=True, exist_ok=False)
        request = {
            "artifacts": artifacts,
            "lib_dir": str(directory),
            "soc_version": self.soc,
            "compile_flags": self.flags,
            "receipt": str(directory / "build.json"),
        }
        (directory / "request.json").write_text(json.dumps(request, indent=2))
        future = self.pool.submit(compile_artifacts, request)
        kernel = DeferredKernel(stage, future)
        stage["kernels"].append(kernel)
        return kernel

    def prepare(self, name, function, args):
        import torch

        stage = {
            "name": name,
            "enabled": False,
            "kernels": [],
            "prepare_start_ns": time.monotonic_ns(),
        }
        self.stages.append(stage)
        self.active = stage
        compiled = torch.compile(
            function, backend="inductor", fullgraph=True, dynamic=False
        )
        try:
            compiled(*args)
        except DeferredLaunch:
            stage["deferred_before_kernel_launch"] = True
        else:
            # External/non-AscendC graphs may have run; never claim overlap for them.
            stage["deferred_before_kernel_launch"] = False
        finally:
            self.active = None
            stage["prepare_end_ns"] = time.monotonic_ns()
        self.check()
        return stage, compiled

    def finish(self, stage, compiled, args):
        self.check()
        stage["wait_start_ns"] = time.monotonic_ns()
        for kernel in stage["kernels"]:
            kernel.future.result(timeout=600)
        stage["enabled"] = True
        out = compiled(*args)
        import torch

        torch.npu.synchronize()
        stage["ready_ns"] = time.monotonic_ns()
        self.check()
        return out

    def report(self):
        return {
            "stages": [
                {
                    **{k: v for k, v in stage.items() if k != "kernels"},
                    "kernels": [
                        {"build": kernel.descriptor, "calls": kernel.calls}
                        for kernel in stage["kernels"]
                    ],
                }
                for stage in self.stages
            ],
            "worker_device_fd_samples": self.observations,
            "worker_observation_errors": self.observation_errors,
            "worker_observation_incomplete": self.observation_incomplete.is_set(),
            "process_exit_or_fd_close_races": self.proc_races,
            "observation_limit": "50ms process-tree FD samples are not syscall-level proof",
        }

    def close(self):
        self.compiler.async_compile = self.original
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.stop.set()
            self.monitor.join(timeout=2)
            if self.monitor.is_alive():
                self._read_error(RuntimeError("monitor did not stop"), None, "shutdown")
