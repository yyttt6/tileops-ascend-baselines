"""CPU regression checks for unsafe acceptance and metadata mixups, not NPU tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from review.common import normalize_params, sha
from review.compare import audit, compare


def row(side, value, pid):
    check = {"checked_outputs": 2, "total_outputs": 2, "unchecked_reason": None}
    return {"schema": "baseline-review-measurement-v1", "side": side,
        "status": "measured_pending_identity_review", "inputs_preserved": True,
        "key": ["ExampleOp", "case"], "device_us": value,
        "payload_sha256": "same-payload", "contract_sha256": "same-contract",
        "protocol": "ascend950-isolated-ffts-v3", "tileops_commit":
        "62c5fa4d1f80b5325c6632aa23622ec2d6c3f2c9" if side == "tileops" else None,
        "full_original_check": check, "measured_output_check": dict(check),
        "timing": {"latency_samples_ms": [value / 1000] * 15,
                   "retained_profiles": [{"profile_path": "/no-such-real-trace",
                       "profile_trace_sha256": "not-real"}] * 15},
        "runtime": {"hostname": "local-machine", "cube": 32, "vector": 64,
            "device_name": "Ascend950", "precision": "original",
            "process_identity": {"pid": pid, "start_ticks": str(pid * 10), "boot_id": "boot"}}}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.paths = [self.directory / n for n in ["b1.json", "b2.json", "d.json"]]
        for p in self.paths:
            p.write_text("{}")
        self.records = {self.paths[0]: row("baseline", 29.5, 1),
                        self.paths[1]: row("baseline", 29.4, 2),
                        self.paths[2]: row("tileops", 10, 3)}
        self.mock = patch("review.compare.audit", side_effect=lambda p, side: copy.deepcopy(self.records[p]))
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def run_compare(self, both=False, identity=False):
        selected = self.paths[:2] if both else self.paths[:1]
        review = None
        if identity:
            review = self.directory / "identity.json"
            review.write_text(json.dumps({"reviewer": "test", "physical_device_identity": "test-device",
                **{k: True for k in ["same_physical_device_driver_firmware", "approved_runtime_profiles",
                    "executing_native_and_jit_binary_identity", "full_api_and_unchanged_precision",
                    "no_interference_or_profiler_loss"]},
                "result_sha256": {str(p.resolve()): sha(p) for p in [*selected, self.paths[2]]}}))
        return compare(selected, self.paths[2], review)

    def test_borderline_requires_new_process(self):
        value = self.run_compare(identity=True)
        self.assertTrue(value["independent_repeat_required"])
        self.assertFalse(value["accepted"])

    def test_independent_repeat_uses_worst_mean(self):
        self.assertTrue(self.run_compare(both=True, identity=True)["accepted"])
        self.records[self.paths[1]]["device_us"] = 30.1
        self.assertFalse(self.run_compare(both=True, identity=True)["numeric_target_met"])

    def test_exact_three_is_not_impossible_but_needs_repeat(self):
        self.records[self.paths[0]]["device_us"] = 30
        self.assertTrue(self.run_compare()["numeric_target_met"])
        self.assertTrue(self.run_compare()["independent_repeat_required"])

    def test_two_files_same_process_are_not_independent(self):
        self.records[self.paths[1]]["runtime"]["process_identity"] = self.records[self.paths[0]]["runtime"]["process_identity"]
        with self.assertRaises(ValueError):
            self.run_compare(both=True)

    def test_fast_number_without_identity_not_accepted(self):
        self.records[self.paths[0]]["device_us"] = 20
        self.assertFalse(self.run_compare()["accepted"])

    def test_cross_host_or_payload_cannot_be_mixed(self):
        for key in ["payload_sha256", "contract_sha256"]:
            old = self.records[self.paths[0]][key]
            self.records[self.paths[0]][key] = "different"
            with self.assertRaises(ValueError):
                self.run_compare()
            self.records[self.paths[0]][key] = old
        self.records[self.paths[0]]["runtime"]["hostname"] = "another-machine"
        with self.assertRaises(ValueError):
            self.run_compare()


class AuditTests(unittest.TestCase):
    def check_row(self, value, replay=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(value))
            return audit(path, "baseline", replay=replay)

    def test_partial_output_is_rejected(self):
        value = row("baseline", 20, 1)
        value["measured_output_check"]["checked_outputs"] = 1
        with self.assertRaises(ValueError):
            self.check_row(value)

    def test_mean_and_all_fifteen_samples_are_required(self):
        value = row("baseline", 20, 1)
        value["device_us"] = 1
        with self.assertRaises(ValueError):
            self.check_row(value)
        value = row("baseline", 20, 1)
        value["timing"]["latency_samples_ms"].pop()
        with self.assertRaises(ValueError):
            self.check_row(value)

    def test_raw_cannot_be_replaced_by_summary(self):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.check_row(row("baseline", 20, 1), replay=True)

    def test_metadata_alias_preserves_contract_and_rejects_collision(self):
        self.assertEqual(normalize_params("TopKMaskFwdOp", {"top_k": [1]}, {"k": [1]}), {"k": [1]})
        with self.assertRaises(ValueError):
            normalize_params("TopKMaskFwdOp", {"top_k": [1], "k": [2]}, {"k": [1]})


if __name__ == "__main__":
    unittest.main()
