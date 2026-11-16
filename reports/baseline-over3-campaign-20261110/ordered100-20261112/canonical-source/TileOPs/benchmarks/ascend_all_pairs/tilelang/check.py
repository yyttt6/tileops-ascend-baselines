"""Reuse the full classic checker without writing its shared provider registry."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT.parent)]
os.environ['TILELANG_CACHE_DIR'] = str(ROOT.parent / '.cache/all-pairs/tilelang/jit')
os.environ['TMPDIR'] = str(ROOT.parent / '.cache/all-pairs/tilelang/tmp')
sys.dont_write_bytecode = True
from scripts import ascend_classic_check as checker
checker.require_guard()
from benchmarks.ascend_all_pairs.tilelang import PROVIDERS
from benchmarks.ops import bench_ascend_classics
bench_ascend_classics.candidate_factories = lambda case: {'tilelang': lambda: PROVIDERS[case._call.signature.name](case)}
raise SystemExit(checker.main())
