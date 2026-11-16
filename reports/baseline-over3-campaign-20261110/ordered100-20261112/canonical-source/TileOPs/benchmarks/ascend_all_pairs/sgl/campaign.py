"""Foreground host orchestration: separate short npu-guard reservations per batch."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--poll',type=float,default=1.0)
    args=p.parse_args()
    queue=json.loads(args.queue.read_text())
    folder=EVIDENCE/'campaigns'/args.name
    folder.mkdir(parents=True,exist_ok=False)
    results=[]
    for i in range(0,len(queue),args.batch_size):
        batch=queue[i:i+args.batch_size]
        name=f'{args.name}_{i//args.batch_size:03d}'
        qfile=folder/f'{name}.json'
        qfile.write_text(json.dumps(batch,indent=2))
        cmd=['npu-guard','run','--owner','allpairs/sgl','--wait','3600','--timeout',str(args.timeout),'--poll',str(args.poll),'--',str(ROOT/'.tools/baselines/baseline-run'),'sgl-kernel-npu','env','PYTHONDONTWRITEBYTECODE=1',f'TRITON_CACHE_DIR={ROOT}/.cache/all-pairs/sgl/triton',f'TORCH_EXTENSIONS_DIR={ROOT}/.cache/all-pairs/sgl/torch','python',str(Path(__file__).with_name('check.py')),'--queue',str(qfile),'--run',name]
        log=folder/f'{name}.log'
        print(json.dumps({'phase':'guard_wait_or_run','batch':name,'count':len(batch)}),flush=True)
        with log.open('w') as stream:
            rc=subprocess.call(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        item={'batch':name,'command':cmd,'exit_code':rc,'log':str(log.relative_to(ROOT)),'queue':str(qfile.relative_to(ROOT)),'finished_at':datetime.now(timezone.utc).isoformat()}
        results.append(item)
        (folder/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(item),flush=True)
        if rc in (75,76,78):
            print('Guard did not grant a clean run; do not bypass it.',flush=True)
            break
    return 0

if __name__=='__main__':raise SystemExit(main())
