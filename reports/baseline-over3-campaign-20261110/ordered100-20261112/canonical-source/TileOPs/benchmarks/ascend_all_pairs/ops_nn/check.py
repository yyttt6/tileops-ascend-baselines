"""Use the unchanged classic checker with the isolated provider dictionary."""
import importlib.util
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent)]
spec = importlib.util.spec_from_file_location("classic_checker", ROOT / "scripts/ascend_classic_check.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.require_guard()
i=sys.argv.index("--output")+1
sys.argv[i]=str(Path(sys.argv[i]).resolve())
os.chdir(ROOT.parent / ".cache/all-pairs/ops_nn")
# Patch this process only; retain all checker oracle and input-protection rules.
from benchmarks.ops import bench_ascend_classics
from benchmarks.ascend_all_pairs.ops_nn.adapters import adaptive_max_pool
PROVIDERS = {"AdaptiveMaxPool2dFwdOp":adaptive_max_pool,"AdaptiveMaxPool2dIndicesFwdOp":adaptive_max_pool}
bench_ascend_classics.candidate_factories = lambda case: {"ops-nn": lambda: PROVIDERS[case._call.signature.name](case)}
capture_path=ROOT.parent / ".agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/native_log_capture.py"
spec2=importlib.util.spec_from_file_location("native_capture",capture_path)
capture=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(capture)
with capture.NativeLogCapture(os.environ["ASCEND_PROCESS_LOG_PATH"],Path(sys.argv[sys.argv.index("--output")+1]).with_suffix(".native.jsonl")):
    rc=checker.main()
raise SystemExit(rc)
