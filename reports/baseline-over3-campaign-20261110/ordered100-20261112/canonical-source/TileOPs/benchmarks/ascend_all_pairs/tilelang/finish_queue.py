"""CPU-only foreground coordinator for bounded, non-nested NPU reservations.

Run this outside npu-guard. Each device batch has its own complete host guard.
No failed receipt is retried and no guard refusal is bypassed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang'
SOURCE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--timeout', type=int, choices=(300, 900), default=300,
                        help='900 is only for a batch requiring original native compilation.')
    parser.add_argument('--compilation-required', action='store_true')
    parser.add_argument('--case-seconds', type=int, default=240)
    args = parser.parse_args()
    if args.timeout == 900 and not args.compilation_required:
        parser.error('900 seconds requires an explicit --compilation-required justification.')
    queue = args.queue.resolve()
    if not queue.is_relative_to(EVIDENCE):
        parser.error('Queue must stay in the assigned TileLang evidence directory.')
    lock = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    if lock.get('phase') == 'running' and lock.get('child', {}).get('pid') == os.getpgrp():
        parser.error('This CPU coordinator must run outside npu-guard; nested reservations are forbidden.')
    rows = json.loads(queue.read_text())
    for row in rows:
        if not (ROOT/row['output']).resolve().is_relative_to(EVIDENCE):
            parser.error('Every output must stay in assigned TileLang evidence.')
    sequence = 0
    while True:
        pending = [r for r in rows if not (ROOT/r['output']).exists()]
        if not pending:
            print(json.dumps({'phase': 'queue_exhausted', 'rows': len(rows)}), flush=True)
            return 0
        sequence += 1
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        log = EVIDENCE/f'continuation-{stamp}-{os.getpid()}-{sequence}.log'
        command = [sys.executable, str(ROOT/'tools/npu_guard.py'), 'run', '--owner',
                   'allpairs/tilelang', '--wait', '3600', '--timeout', str(args.timeout), '--',
                   str(ROOT/'.tools/baselines/baseline-run'), 'tilelang', 'python', '-B',
                   str(SOURCE/'batch_traced.py'), '--queue', str(queue),
                   '--seconds', str(args.timeout - (100 if args.timeout == 900 else 60)),
                   '--case-seconds', str(args.case_seconds)]
        state = {'schema': 'tilelang-foreground-continuation/v1', 'phase': 'reservation',
                 'queue': str(queue), 'pending_cases': len(pending), 'sequence': sequence,
                 'started_at': datetime.now(timezone.utc).isoformat(), 'log': str(log),
                 'command': command, 'guard_wait_refusal_policy': 'stop; no retry or bypass'}
        state_path = EVIDENCE/'continuation_state.json'
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({k: state[k] for k in ('phase', 'sequence', 'pending_cases', 'log')}), flush=True)
        with log.open('x') as output:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                for line in child.stdout:
                    output.write(line)
                    output.flush()
                    if line.startswith('{'):
                        try:
                            result = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if 'status' in result and 'case_id' in result:
                            print(json.dumps({k: result.get(k) for k in
                                              ('op', 'case_id', 'status', 'wall_seconds')}), flush=True)
                code = child.wait()
        remaining = sum(not (ROOT/r['output']).exists() for r in rows)
        if code == 76:
            # The host protocol invalidates this complete contaminated batch.
            invalid_path = EVIDENCE/'invalidated_batches.json'
            invalid = json.loads(invalid_path.read_text()) if invalid_path.exists() else {
                'schema': 'tilelang-invalidated-batches/v1', 'batches': []}
            invalid['batches'].append({'guard_exit': code, 'log': str(log.relative_to(ROOT)),
                                       'started_at': state['started_at'],
                                       'reason': 'Host guard detected an external NPU process; entire batch is excluded.',
                                       'receipts': [r['output'] for r in pending if (ROOT/r['output']).exists()]})
            invalid_path.write_text(json.dumps(invalid, indent=2)+'\n')
        state.update(phase='reservation_finished', guard_exit=code,
                     finished_at=datetime.now(timezone.utc).isoformat(), remaining_cases=remaining)
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'guard_exit': code, 'remaining_cases': remaining}), flush=True)
        for report in ('artifacts.py', 'report.py'):
            subprocess.run([sys.executable, '-B', str(SOURCE/report)], cwd=ROOT, check=True)
        if code:
            return code
        if remaining == len(pending):
            print('No complete case fit this reservation; stop for an explicit queue/budget correction.', flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
