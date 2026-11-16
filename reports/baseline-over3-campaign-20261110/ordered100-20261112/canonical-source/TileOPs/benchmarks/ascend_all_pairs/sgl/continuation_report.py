"""CPU-only concise live results; retain each original receipt and attempt."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl'
rows=[]
attempts=[]
for p in sorted((EVIDENCE/'campaigns').glob('continuation*/results.json')):
    attempts.extend(json.loads(p.read_text()))
attempt_by_batch={r['batch']:r for r in attempts}
for p in sorted((EVIDENCE/'receipts').glob('continuation*/*.json')):
    r=json.loads(p.read_text())
    drift=sorted({f for name in ('source_sha256','cpu_prepared_source_sha256','oracle_source_sha256','native_trace_source_sha256')
                  for f,expected in r.get(name,{}).items()
                  if not (ROOT/f).exists() or hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=expected})
    attempt=attempt_by_batch.get(p.parent.name)
    error=r.get('error','')
    if len(error)>1200:
        useful=[line for line in error.splitlines() if any(w in line.lower() for w in ['error:','overflow','failed','unsupported'])]
        error='\n'.join(useful[-5:]) if useful else error[:1200]
    status = 'held_for_original_NPU_input_rerun' if r.get('cpu_prepared_artifact') or r.get('input_generation_device') == 'cpu' else r['status']
    rows.append({'op':r['op'],'case_id':r['case_id'],'status':status,'original_receipt_status':r['status'],'stage':r.get('stage'),'error_type':r.get('error_type'),
        'error':error,'reason':r.get('reason'),'wall_seconds':r.get('wall_seconds'),'finished_at':r.get('finished_at'),
        'receipt':str(p.relative_to(ROOT)),'source_mismatches':drift,'verification':r.get('verification'),
        'candidate_calls':r.get('candidate_calls'),'native_dispatch':r.get('runtime_native_calls'),
        'triton_launches':len(r.get('native_kernel_launches',[])),'attribution_complete':r.get('attribution_complete'),
        'cpu_prepared':bool(r.get('cpu_prepared_artifact')),
        'guard_completed':attempt is not None,'guard_exit':None if attempt is None else attempt['exit_code'],
        'guard_uncontaminated':attempt is not None and attempt['exit_code'] != 76})
latest={}
for row in sorted(rows,key=lambda r:r.get('finished_at') or ''):latest[row['op'],row['case_id']]=row
summary={'updated_at':datetime.now(timezone.utc).isoformat(),'receipts':len(rows),'unique_cases':len(latest),
    'latest_status_counts':dict(Counter(r['status'] for r in latest.values())),
    'native_verified_current':sum(r['status']=='verified' and r['attribution_complete'] and not r['source_mismatches'] and r['guard_uncontaminated'] for r in latest.values()),
    'provisional_verified_running_guard':sum(r['status']=='verified' and not r['guard_completed'] for r in latest.values()),
    'successful_ops':sorted({r['op'] for r in latest.values() if r['status']=='verified' and not r['source_mismatches'] and r['guard_uncontaminated']}),
    'failed_ops':sorted({r['op'] for r in latest.values() if r['status'] not in ('verified','unsupported','held_for_original_NPU_input_rerun')}),
    'cpu_prepared_used':sum(r['cpu_prepared'] for r in rows),'completed_guard_attempts':len(attempts),
    'attempt_exit_counts':dict(Counter(r['exit_code'] for r in attempts))}
(EVIDENCE/'continuation_20261021T1415/live_results.json').write_text(json.dumps({'summary':summary,'latest':list(latest.values()),'receipts':rows,'attempts':attempts},indent=2))
print(json.dumps(summary,indent=2))
