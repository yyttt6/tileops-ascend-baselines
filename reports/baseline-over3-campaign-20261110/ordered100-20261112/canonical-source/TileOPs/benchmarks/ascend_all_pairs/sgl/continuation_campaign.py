"""One pending FIFO reservation; immutable attempts and time-bounded batches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl_original_input_recovery'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--hours', type=float, default=6)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=48)
    args = p.parse_args()
    if not 0 < args.seconds <= 60 or not 0 < args.timeout <= 300:
        p.error('Recovery uses complete-case soft <=60 seconds and guard hard <=300 seconds.')
    pending = json.loads(args.queue.read_text())
    folder = EVIDENCE / 'campaigns' / args.name
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.monotonic()
    consecutive_not_started = 0
    while pending and time.monotonic() - started < args.hours * 3600:
        name = f'{args.name}_{len(results):04d}'
        batch = pending[:args.batch_size]
        qfile = folder / f'{name}.json'
        qfile.write_text(json.dumps(batch, indent=2))
        cmd = ['/usr/local/bin/npu-guard','run','--owner','allpairs/sgl-original-input-recovery','--wait','7200','--timeout',str(args.timeout),'--poll','0.1','--',str(ROOT/'.tools/baselines/baseline-run'),'sgl-kernel-npu','env','PYTHONDONTWRITEBYTECODE=1',f'TRITON_CACHE_DIR={ROOT}/.cache/all-pairs/sgl/triton',f'TORCH_EXTENSIONS_DIR={ROOT}/.cache/all-pairs/sgl/torch','python',str(Path(__file__).with_name('continuation_check.py')),'--queue',str(qfile),'--run',name,'--seconds',str(args.seconds)]
        log = folder / f'{name}.log'
        requested = datetime.now(timezone.utc).isoformat()
        print(json.dumps({'phase':'guard_wait_or_run','batch':name,'pending':len(pending),'requested_at':requested}), flush=True)
        with log.open('w') as stream:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        starts = []
        for line in log.read_text(errors='replace').splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record,dict) and record.get('phase') == 'case_start':
                starts.append((record['op'],record['case_id']))
        receipt_dir = EVIDENCE / 'receipts' / name
        receipts = [json.loads(f.read_text()) for f in receipt_dir.glob('*.json')] if receipt_dir.exists() else []
        completed = {(r['op'],r['case_id']) for r in receipts}
        # Started cases lacking receipts retain timeout/crash evidence; do not
        # repeatedly execute the same fault while processing the remaining list.
        attempted = completed | set(starts)
        if rc == 76:
            attempted = set()  # contaminated receipts are never accepted
        pending = [r for r in pending if (r['op'],r['case_id']) not in attempted]
        item = {'batch':name,'command':cmd,'exit_code':rc,'log':str(log.relative_to(ROOT)),'queue':str(qfile.relative_to(ROOT)),'requested_at':requested,'finished_at':datetime.now(timezone.utc).isoformat(),'started':len(starts),'receipts':len(receipts),'pending':len(pending)}
        results.append(item)
        (folder/'results.json').write_text(json.dumps(results,indent=2))
        (folder/'pending.json').write_text(json.dumps(pending,indent=2))
        print(json.dumps(item),flush=True)
        consecutive_not_started = consecutive_not_started + 1 if rc == 75 and not attempted else 0
        if consecutive_not_started >= 2:
            print('Two terminal guard75 attempts without device start; pending CPU todo retained.',flush=True)
            break
        if rc == 78 or rc == 76:
            print('Device health/contamination requires review before another reservation.',flush=True)
            break
        if not attempted and rc != 75:
            print('No case started; inspect deployment failure before retry.',flush=True)
            break
    print(json.dumps({'phase':'campaign_finished','pending':len(pending),'attempts':len(results)}),flush=True)


if __name__ == '__main__':
    main()
