#!/usr/bin/env python3
"""Run a bounded list of exact cases sequentially in one foreground reservation."""
import json, pathlib, runpy, sys, time
root = pathlib.Path(__file__).resolve().parents[3].parent
queue = json.loads(pathlib.Path(sys.argv[1]).read_text())
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 780.0
started = time.monotonic()
for item in queue:
    if time.monotonic() - started > budget:
        print('BATCH_BUDGET_REACHED', flush=True)
        break
    output = root / item['receipt']
    if output.exists():
        continue
    sys.argv = ['check.py', '--op', item['op'], '--case-id', item['case_id'], '--library', 'TileKernels', '--output', str(output)]
    try:
        runpy.run_path(str(pathlib.Path(__file__).with_name('check.py')), run_name='__main__')
    except SystemExit as exc:
        print('CASE_EXIT', item['op'], item['case_id'], exc.code, flush=True)
    if output.exists():
        row = json.loads(output.read_text())
        if row.get('status') in ('device_sync_failed', 'cleanup_failed'):
            break
