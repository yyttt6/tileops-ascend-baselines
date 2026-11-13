"""CPU-only delivery checks; no torch import, driver access or NPU work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import materialize_patch
from controller import launch_owned
from watchdog import supervise

P = Path(__file__).resolve().parent

class Delivery(unittest.TestCase):
    def test_reviewed_manifest(self):
        manifest = P / 'MANIFEST.json'
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         'deafc7c57a9bdd0968bd9b1ed5465a36a1a3b2bc748a96a9561beeb1188f113b')
        for name, digest in json.loads(manifest.read_text())['files'].items():
            self.assertEqual(hashlib.sha256((P/name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(len(json.loads((P/'source-pins.json').read_text())), 589)

    def test_entry_help_without_dependencies(self):
        for entry in ['controller.py', 'pilot.py', 'environment_entry.py',
                      'prepare_envs.py', 'materialize_patch.py']:
            subprocess.run([sys.executable, '-S', str(P/entry), '--help'],
                           stdout=subprocess.DEVNULL, check=True, timeout=10)

    def test_owned_child_reaped(self):
        receipt = {}
        self.assertEqual(launch_owned([sys.executable, '-c', 'pass'], [], receipt), 0)
        self.assertTrue(receipt['reaped'])

    def test_watchdog_stops_only_test_child(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'receipt'
            rc = supervise([sys.executable, '-c', 'import time;time.sleep(20)'],
                           output, timeout=.2, grace=.2, reap=.2)
            self.assertNotEqual(rc, 0)
            self.assertEqual(json.loads((output/'watchdog-receipt.json').read_text())['remaining_live'], [])

    def test_reconstructed_core_imports_without_site_packages(self):
        from prepare_sources import prepare
        with tempfile.TemporaryDirectory() as td:
            frozen = Path(td)/'frozen'
            self.assertEqual(prepare(P.parents[1], frozen), 592)
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from tools import kernel_task_profile_full32 as parser; "
                "sys.modules['tools.kernel_task_profile'] = parser; "
                "from tools import measure_baselines; "
                "assert 'torch' not in sys.modules; "
                "assert 'torch_npu' not in sys.modules"
            )
            subprocess.run([sys.executable, '-S', '-c', code, str(frozen)],
                           check=True, timeout=10)

    def test_wrong_patch_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td)/'original', Path(td)/'candidate'
            source.write_bytes(b'not the reviewed backend')
            with self.assertRaises(ValueError):
                materialize_patch.materialize(source, output)
            self.assertFalse(output.exists())

if __name__ == '__main__':
    unittest.main()
