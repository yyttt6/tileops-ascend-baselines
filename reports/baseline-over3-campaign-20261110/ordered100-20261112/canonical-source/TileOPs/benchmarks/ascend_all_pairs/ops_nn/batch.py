"""Guarded foreground validation batches; preserve the shared checker unchanged."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];WS=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(WS)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=840);args=p.parse_args()
    args.jobs=args.jobs.resolve();args.out=args.out.resolve()
    os.chdir(WS/'.cache/all-pairs/ops_nn')
    os.environ['ASCEND_PROCESS_LOG_PATH']=str(args.out.resolve()/'cann_logs')
    Path(os.environ['ASCEND_PROCESS_LOG_PATH']).mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('shared_checker',ROOT/'scripts/ascend_classic_check.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);checker.require_guard()
    capture_path=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/native_log_capture.py'
    spec2=importlib.util.spec_from_file_location('native_capture',capture_path);capture=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(capture)
    from benchmarks.ops import bench_ascend_classics
    from benchmarks.ascend_all_pairs.ops_nn.staging import CANDIDATES
    from benchmarks.ascend_all_pairs.ops_nn.capture_cursors import seed_capture
    bench_ascend_classics.candidate_factories=lambda case:{'ops-nn':lambda:CANDIDATES[case._call.signature.name](case)}
    args.out.mkdir(parents=True,exist_ok=True);outcomes=[];start=time.monotonic();capture_state={}
    sources=[Path(__file__),Path(__file__).with_name('capture_cursors.py'),capture_path]
    (args.out/'capture_sources.json').write_text(json.dumps({str(f.relative_to(WS)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},indent=2)+'\n')
    with capture.NativeLogCapture(os.environ['ASCEND_PROCESS_LOG_PATH'],args.out/'all-native.jsonl'):
        for i,job in enumerate(json.loads(args.jobs.read_text())):
            if (args.out/'STOP').exists():
                print('Requested stop at synchronized case boundary; unstarted cases remain unverified.',flush=True);break
            if time.monotonic()-start>=args.seconds:
                print('Batch time budget reached; unstarted cases remain unverified.',flush=True);break
            receipt=args.out/f'{i:03d}-{job["op"]}.json'
            if receipt.exists():raise FileExistsError(receipt)
            print(json.dumps({'starting':job,'elapsed':time.monotonic()-start}),flush=True)
            sys.argv=[str(ROOT/'scripts/ascend_classic_check.py'),'--op',job['op'],'--case-id',job['case_id'],'--library','ops-nn','--output',str(receipt)]
            cap=capture.NativeLogCapture(os.environ['ASCEND_PROCESS_LOG_PATH'],args.out/f'{i:03d}-native.jsonl')
            cap.state=seed_capture(os.environ['ASCEND_PROCESS_LOG_PATH'],capture_state)
            with cap:
                rc=checker.main()
            capture_state=dict(cap.state)
            row=json.loads(receipt.read_text());outcomes.append(dict(job=job,receipt=str(receipt.relative_to(WS)),status=row['status'],exit_code=rc))
            (args.out/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
            errors=' '.join(str(row.get(k,'')) for k in ('error','sync_error','cleanup_error'))
            if rc==78 or any(t in errors for t in ('507035','507014','vector core exception')):
                print('Device fault: remaining cases not executed.',flush=True);return 78
            gc.collect()
            import torch
            torch.npu.synchronize();torch.npu.empty_cache()
        time.sleep(1.0)  # Drain delayed CANN log writes after the final synchronized case.
    return 0
if __name__=='__main__':raise SystemExit(main())
