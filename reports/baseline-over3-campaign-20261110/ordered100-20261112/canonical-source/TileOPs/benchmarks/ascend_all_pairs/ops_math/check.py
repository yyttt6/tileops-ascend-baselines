#!/usr/bin/env python3
"""Use the unchanged common checker and numerical policies with this provider."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT.parent)]
spec=importlib.util.spec_from_file_location('original_checker',ROOT/'scripts/ascend_classic_check.py')
checker=importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
checker.require_guard()
# Runtime imports happen only after require_guard.
from benchmarks.ascend_all_pairs.ops_math import CANDIDATES
import benchmarks.ops.bench_ascend_classics as registry
registry.candidate_factories=lambda case: {'ops-math':lambda:CANDIDATES[case._call.signature.name](case)}
raise SystemExit(checker.main())
