#!/usr/bin/env python3
"""Run bounded jobs with the original checker; invoked inside one npu-guard."""
import argparse,importlib.util,json,os,sys,gc,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
# The adjacent CLI queue.py must never shadow Python's stdlib queue when
# Torch imports multiprocessing. Provider imports use the full package path.
own_directory=Path(__file__).resolve().parent
sys.path[:]=[str(ROOT),str(ROOT/'src'),str(ROOT.parent)]+[s for s in sys.path if Path(s or '.').resolve()!=own_directory]
p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
# Preserve already queued foreground commands, but remove the full source-only
# ACLNN package before Torch/ld.so initialize. Its host API otherwise dispatches
# installed built-in binaries with a different tiling ABI during verification.
full_vendor=ROOT.parent/'.cache/all-pairs/ops_math/install/vendors/allpairs_math_math'
graph_vendor=ROOT.parent/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math'
if graph_vendor.exists() and os.environ.get('OPS_MATH_CHECKER_RUNTIME_READY')!='1':
 lock=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
 if lock.get('phase')!='running' or lock.get('child',{}).get('pid')!=os.getpgrp():raise RuntimeError('Host guard required before runtime environment restart.')
 existing=[s for s in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if s and s not in (str(full_vendor),str(graph_vendor))]
 os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(existing)
 os.environ['LD_LIBRARY_PATH']=':'.join(s for s in os.environ.get('LD_LIBRARY_PATH','').split(':') if s and not any(s.startswith(str(v)) for v in (full_vendor,graph_vendor)))
 os.environ['OPS_MATH_CHECKER_RUNTIME_READY']='1'
 os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
a.output_dir.mkdir(parents=True,exist_ok=True)
console=open(a.output_dir/'console.log','a',buffering=1)
os.dup2(console.fileno(),1);os.dup2(console.fileno(),2)
os.environ['ASCEND_PROCESS_LOG_PATH']=str(a.output_dir/'cann-logs')
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
os.environ['OPS_MATH_GRAPH_CALLS']=str(a.output_dir/'graph-calls.jsonl')
os.environ['OPS_MATH_NATIVE_CHILD_RECORDS']=str(a.output_dir/'native-children')
cache=ROOT.parent/'.cache/all-pairs/ops_math'
for key,leaf in [('ASCEND_CACHE_PATH','runtime-cache'),('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_EXTENSIONS_DIR','torch_extensions')]:
 d=cache/leaf;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
os.chdir(cache)
loader=importlib.util.spec_from_file_location('common_checker',ROOT/'scripts/ascend_pair_check.py'); checker=importlib.util.module_from_spec(loader);loader.loader.exec_module(checker);checker.require_guard()
# Capture stream while logs rotate; no independent device work or background processes.
logfile=ROOT.parent/'.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/native_log_capture.py'
spec=importlib.util.spec_from_file_location('native_log_capture',logfile);logs=importlib.util.module_from_spec(spec);spec.loader.exec_module(logs)
from benchmarks.ascend_all_pairs.ops_math import CANDIDATES
import benchmarks.ascend_all_pairs.registry as registry
def private_factory(case,library):
 if library!='ops-math':raise ValueError(library)
 return lambda:CANDIDATES[case._call.signature.name](case)
registry.candidate_factory=private_factory
import torch
results=[];device_failed=False
jobs=json.loads(a.jobs.read_text())
command_file=a.output_dir/'command.json'
command=json.loads(command_file.read_text()) if command_file.exists() else {}
# Stop between complete cases, leaving time for log capture and synchronization.
# The host guard still bounds a single slow case or its original JIT.
budget=float(command.get('soft_budget_seconds',60))
batch_started=time.monotonic()
with logs.NativeLogCapture(a.output_dir/'cann-logs',a.output_dir/'native-events.jsonl'):
 for n,job in enumerate(jobs):
  if results and time.monotonic()-batch_started>=budget:break
  out=a.output_dir/f'{n:03d}.json'
  active={'index':n,'job':job,'phase':'running','started_at':datetime.now(timezone.utc).isoformat(),'receipt':str(out)}
  (a.output_dir/'active-case.json').write_text(json.dumps(active,indent=2)+'\n')
  sys.argv=['ascend_pair_check.py','--op',job['op'],'--case-id',job['case_id'],'--library','ops-math','--output',str(out)]
  rc=checker.main(); results.append({'job':job,'receipt':str(out),'exit_code':rc})
  active.update(phase='complete',exit_code=rc,finished_at=datetime.now(timezone.utc).isoformat())
  (a.output_dir/'active-case.json').write_text(json.dumps(active,indent=2)+'\n')
  (a.output_dir/'results.json').write_text(json.dumps(results,indent=2))
  if rc==78 or os.environ.pop('OPS_MATH_CHILD_DEVICE_FAILED','')=='1':device_failed=True;break
  gc.collect();torch.npu.empty_cache()
 if not device_failed:torch.npu.synchronize()
(a.output_dir/'batch_summary.json').write_text(json.dumps({'completed_jobs':len(results),'total_jobs':len(jobs),'remaining_jobs':jobs[len(results):],'device_failed':device_failed,'elapsed_seconds':time.monotonic()-batch_started,'between_case_budget_seconds':budget},indent=2)+'\n')
print(json.dumps({'completed':len(results),'jobs':str(a.jobs)}),flush=True)
if device_failed:raise SystemExit(78)
