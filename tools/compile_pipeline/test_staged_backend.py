import concurrent.futures
import errno
import gc
import hashlib
import threading
import weakref
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from run_validation import measure_and_release

from staged_backend import DeferredKernel, DeferredLaunch, StagedBackend


class StagingTests(unittest.TestCase):
    def test_deferred_launch_never_waits_or_executes(self):
        future = concurrent.futures.Future()
        kernel = DeferredKernel({"name": "next", "enabled": False}, future)
        with self.assertRaises(DeferredLaunch):
            kernel(1)
        self.assertFalse(future.done())
        self.assertEqual(kernel.calls, 0)

    def test_consumption_uses_worker_artifact_and_reuses_loaded_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.so"
            path.write_bytes(b"worker-result")
            built = []
            called = []

            class Kernel:
                def __init__(self, *args):
                    built.append(args)

                def __call__(self, *args):
                    called.append(args)
                    return 9

            mod = types.ModuleType("mock_kernel")
            mod.NpuInductorKernel = Kernel
            auto = types.ModuleType("mock_autotune")
            auto.NpuInductorAutoTuneKernel = Kernel
            auto.KernelInfo = lambda **x: x
            future = concurrent.futures.Future()
            future.set_result(
                {
                    "lib_wrapper": "wrapper.so",
                    "lib_kernel": str(path),
                    "kernel_key": "key",
                    "artifact_hashes": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    },
                }
            )
            kernel = DeferredKernel({"name": "next", "enabled": True}, future)
            with patch.dict(
                "sys.modules",
                {
                    "torch_npu._inductor.ascendc.compiler._kernel": mod,
                    "torch_npu._inductor.ascendc.compiler._autotune": auto,
                },
            ):
                self.assertEqual(kernel(1), 9)
                self.assertEqual(kernel(2), 9)
            self.assertEqual(built, [("wrapper.so", str(path), "key")])
            self.assertEqual(called, [(1,), (2,)])
            self.assertEqual(kernel.calls, 2)

    def test_changed_artifact_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.so"
            path.write_bytes(b"changed")
            mod = types.ModuleType("mock_kernel")
            mod.NpuInductorKernel = lambda *x: self.fail("must not load")
            auto = types.ModuleType("mock_autotune")
            auto.NpuInductorAutoTuneKernel = None
            auto.KernelInfo = None
            future = concurrent.futures.Future()
            future.set_result({"artifact_hashes": {str(path): "wrong"}})
            kernel = DeferredKernel({"name": "next", "enabled": True}, future)
            with patch.dict(
                "sys.modules",
                {
                    "torch_npu._inductor.ascendc.compiler._kernel": mod,
                    "torch_npu._inductor.ascendc.compiler._autotune": auto,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                    kernel(1)

    def test_late_compilation_is_not_silently_allowed(self):
        backend = object.__new__(StagedBackend)
        backend.active = None
        with self.assertRaisesRegex(RuntimeError, "outside graph preparation"):
            backend._compile(None, {})


class LifetimeTests(unittest.TestCase):
    def test_completed_cases_do_not_retain_inputs_or_callable(self):
        class Tensor:
            pass

        items = []
        references = []
        for _ in range(100):
            tensor = Tensor()
            item = {"args": (tensor,), "compiled": lambda x=tensor: x}
            references.append(weakref.ref(tensor))
            del tensor
            items.append(item)
            measure_and_release(item, lambda current: current["compiled"]())
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertTrue(all(not item for item in items))

    def test_failure_releases_inputs_but_preserves_next_case(self):
        first = {"args": object(), "compiled": object()}
        next_case = {"args": object(), "compiled": object()}
        saved = dict(next_case)

        def fail(item):
            raise RuntimeError("measurement failed")

        with self.assertRaisesRegex(RuntimeError, "measurement failed"):
            measure_and_release(first, fail)
        self.assertEqual(first, {})
        self.assertEqual(next_case, saved)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(StagedBackend)
        self.backend.pool = types.SimpleNamespace(_processes={123: object()})
        self.backend.observations = []
        self.backend.observation_errors = []
        self.backend.observation_incomplete = threading.Event()
        self.backend.unsafe = threading.Event()
        self.backend.proc_races = 0

    def test_process_tree_permission_failure_invalidates(self):
        with patch.object(
            Path, "read_text", side_effect=PermissionError(errno.EACCES, "denied")
        ):
            self.backend._sample_workers(set())
        with self.assertRaisesRegex(RuntimeError, "observation incomplete"):
            self.backend.check()
        self.assertEqual(self.backend.observation_errors[0]["pid"], 123)

    def test_fd_permission_failure_invalidates(self):
        with (
            patch.object(Path, "read_text", return_value=""),
            patch.object(Path, "iterdir", return_value=iter([Path("/proc/123/fd/4")])),
            patch("os.readlink", side_effect=PermissionError(errno.EACCES, "denied")),
        ):
            self.backend._sample_workers(set())
        with self.assertRaisesRegex(RuntimeError, "observation incomplete"):
            self.backend.check()

    def test_exited_process_and_closed_fd_are_recorded_races(self):
        for code in (errno.ENOENT, errno.ESRCH):
            self.backend._read_error(OSError(code, "gone"), 123, "read")
        self.backend.check()
        self.assertEqual(self.backend.proc_races, 2)
        self.assertEqual(self.backend.observation_errors, [])

    def test_unexpected_monitor_failure_invalidates(self):
        self.backend.stop = types.SimpleNamespace(wait=lambda interval: False)
        with patch.object(
            self.backend, "_sample_workers", side_effect=RuntimeError("thread failed")
        ):
            self.backend._monitor()
        with self.assertRaisesRegex(RuntimeError, "observation incomplete"):
            self.backend.check()


if __name__ == "__main__":
    unittest.main()
