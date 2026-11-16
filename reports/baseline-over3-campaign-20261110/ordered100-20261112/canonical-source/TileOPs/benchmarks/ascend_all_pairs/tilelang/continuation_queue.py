"""One foreground reservation at a time; CPU replay after terminal guard exit."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang_continuation'
CACHE = ROOT/'.cache/all-pairs/tilelang_continuation'
RUNNER = Path(__file__).with_name('continuation.py')
COORDINATOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dump(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')


def main():
    pending = json.loads(Path(sys.argv[1]).read_text())
    campaign = sys.argv[2]
    (OUT/'reservations').mkdir(exist_ok=True)
    index = 0
    infra_retries = 0
    while pending:
        index += 1
        tag = f'{campaign}-{index:03d}'
        queue = OUT/'reservations'/(tag+'.queue.json')
        # A fresh tag for each attempt; guard kill cannot recycle an old result.
        rows = [{**r, 'tag': f'{tag}-{i:03d}-{r["op"]}-{r["case_id"]}'}
                for i, r in enumerate(pending)]
        dump(queue, rows)
        command = ['npu-guard','run','--owner','allpairs/tilelang-continuation',
                   '--wait','7200','--timeout','300','--',
                   str(ROOT/'.tools/baselines/baseline-run'),'tilelang','python','-B',str(RUNNER),
                   'device','--queue',str(queue)]
        started = datetime.now(timezone.utc).isoformat()
        with (OUT/'reservations'/(tag+'.log')).open('x') as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = {'tag':tag,'command':command,'started_at':started,
                  'finished_at':datetime.now(timezone.utc).isoformat(),'guard_exit':proc.returncode,
                  'coordinator_source_sha256':COORDINATOR_SOURCE_SHA256,
                  'workload_launched': False if proc.returncode == 75 else
                    True if any((CACHE/'runs'/r['tag']/'capture.json').exists() for r in rows) else None}
        dump(OUT/'reservations'/(tag+'.result.json'), result)
        print(json.dumps(result),flush=True)
        if proc.returncode == 75:
            if infra_retries < 1:
                infra_retries += 1
                continue
            break
        captured = [r for r in rows if (CACHE/'runs'/r['tag']/'capture.json').is_file()]
        # Independent CPU-only replays may overlap after the guard terminates.
        # Large complete packets stay serial to bound host memory use.
        workers = 4 if all(not (CACHE/'runs'/r['tag']/'packet.pt').exists() or
                           (CACHE/'runs'/r['tag']/'packet.pt').stat().st_size < 512 * 1024**2
                           for r in captured) else 1
        def compare(row):
            run = CACHE/'runs'/row['tag']
            # Even failed captures get immutable CPU disposition; never call
            # them numerical failures merely because a guard did not start.
            command = ['/usr/bin/python3','-I','-S','-B',str(RUNNER),'compare','--run',str(run)]
            with (run/'compare.log').open('x') as log:
                checked = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            return row, checked.returncode
        completed = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row, code in pool.map(compare, captured):
                completed.add((row['op'],row['case_id']))
                print(json.dumps({'tag':row['tag'],'cpu_compare_exit':code,
                                  'cpu_replay_workers':workers}),flush=True)
        pending = [r for r in pending if (r['op'],r['case_id']) not in completed]
        dump(OUT/'reservations'/(tag+'.remaining.json'),pending)
        if proc.returncode != 0 or not completed:
            break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
