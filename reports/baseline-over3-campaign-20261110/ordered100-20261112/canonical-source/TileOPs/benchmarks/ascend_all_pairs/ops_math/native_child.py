#!/usr/bin/env python3
"""One original native request, foreground inside its parent's host reservation."""
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

own = Path(__file__).resolve().parent
root = own.parents[3]
sys.path[:] = [str(root / 'TileOPs'), str(root / 'TileOPs/src'), str(root)] + [p for p in sys.path if Path(p or '.').resolve() != own]
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--record', type=Path, required=True)
args = parser.parse_args()
from benchmarks.ascend_all_pairs.ops_math.isolated import require_guard
require_guard()
record = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
          'process_group': os.getpgrp(), 'parent_pid': os.getppid(),
          'status': 'preparing', 'custom_opp': os.environ.get('ASCEND_CUSTOM_OPP_PATH')}

try:
    import torch
    import torch_npu
    torch.npu.set_device(0)
    payload = torch.load(args.input, map_location='cpu', weights_only=False)

    def map_tensors(value, fn):
        if isinstance(value, torch.Tensor):
            return fn(value)
        if isinstance(value, tuple):
            return tuple(map_tensors(v, fn) for v in value)
        if isinstance(value, list):
            return [map_tensors(v, fn) for v in value]
        if isinstance(value, dict):
            return {k: map_tensors(v, fn) for k, v in value.items()}
        return value

    inputs = map_tensors(payload['inputs'], lambda t: t.to('npu'))
    case = SimpleNamespace(id=payload['case_id'], params=payload['params'], inputs=inputs)
    from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
    candidate = NATIVE_CANDIDATES[payload['op']](case)
    record.update(op=payload['op'], case_id=payload['case_id'], api=candidate.api,
                  source_sha256=candidate.source_sha256, status='native_execution_running')
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    output = candidate.implementation.run(*inputs)
    torch.npu.synchronize()
    host_output = map_tensors(output, lambda t: t.detach().cpu())
    if candidate.cleanup is not None:
        candidate.cleanup()
    torch.npu.synchronize()
    torch.save(host_output, args.output)
    record.update(status='native_execution_complete', finished_at=datetime.now(timezone.utc).isoformat())
    args.record.write_text(json.dumps(record, indent=2) + '\n')
except BaseException as error:
    record.update(status='native_execution_failed', error=repr(error), error_type=type(error).__name__, traceback=traceback.format_exc(),
                  finished_at=datetime.now(timezone.utc).isoformat())
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    if '507035' in str(error):
        raise SystemExit(78)
    raise
