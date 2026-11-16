"""Foreground bounded batch. Caller must hold npu-guard for the entire process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT.parent)]
sys.dont_write_bytecode = True
os.environ['TILELANG_CACHE_DIR'] = str(ROOT.parent / '.cache/all-pairs/tilelang/jit')
os.environ['TMPDIR'] = str(ROOT.parent / '.cache/all-pairs/tilelang/tmp')
from scripts import ascend_classic_check as checker
checker.require_guard()
p = argparse.ArgumentParser()
p.add_argument('--queue', type=Path, required=True)
p.add_argument('--seconds', type=int, default=240)
args = p.parse_args()
from benchmarks.ascend_all_pairs.tilelang import PROVIDERS
from benchmarks.ascend_all_pairs.tilelang.diagnostics import DIAGNOSTIC_PROVIDERS
from benchmarks.ops import bench_ascend_classics
bench_ascend_classics.candidate_factories = lambda case: {'tilelang': lambda: (DIAGNOSTIC_PROVIDERS | PROVIDERS)[case._call.signature.name](case)}
started = time.monotonic()
for row in json.loads(args.queue.read_text()):
    if time.monotonic() - started > args.seconds:
        break
    output = ROOT.parent / row['output']
    if output.exists():
        continue
    sys.argv = ['check', '--op', row['op'], '--case-id', row['case_id'], '--library', 'tilelang', '--output', str(output)]
    checker.main()
    # The checker synchronizes each case. Garbage collection frees old inputs.
    import gc
    import torch
    gc.collect()
    torch.npu.empty_cache()
