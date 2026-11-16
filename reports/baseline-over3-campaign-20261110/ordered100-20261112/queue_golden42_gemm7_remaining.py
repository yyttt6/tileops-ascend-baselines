"""Finite seven Gemm continuation under newly certified golden; no numeric retry."""
from pathlib import Path
import datetime, json, os, sys, fcntl, time, hashlib, subprocess
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import rollout_controller as c
from summarize_golden42_gemm7 import summarize

now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scope=[487,489,491,492,493,495,498]
state=D/'GOLDEN42_GEMM7_CONTINUATION.json'
record={'pid':os.getpid(),'entry':Path(__file__).name,'started_at':now(),'scope':scope,'host':c.H,'status':'preflight'}
def save(**kw):
    record.update(kw,checked_at=now());c.put(state,record)
    c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'phase':record['status'],'ordinal':record.get('ordinal'),'checked_at':now()})
def active():
    return not (D/'STOP').exists() and not (D.parent/'STOP').exists() and json.loads((D.parent/'STATE.json').read_text())['status']=='active'

with (D/'golden42-gemm7-continuation.lock').open('a') as own:
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cert=json.loads((D/'GOLDEN42_CERTIFICATES.json').read_text())
    assert cert['status']=='all42_certified'
    save(status='fresh_H1_admission_pending')
    try:
        for i in scope:
            accepted=set(json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
            if i in accepted:continue
            assert not list(D.glob(f'rollout-golden42-nn-gemm7-{i}-golden42-ownes-runtime-h[12]-v1.CHECKPOINT.json')),'Do not repeat completed new-golden run'
            assert not list(D.glob(f'rollout-golden42-nn-gemm7-{i}-golden42-ownes-runtime-h[12]-v1.FAILED.json')),'Preserve failure; diagnose rather than retry'
            while True:
                if not active():raise RuntimeError('User STOP/pause')
                with (D/'controller.lock').open('a') as slot:
                    fcntl.flock(slot,fcntl.LOCK_EX)
                    if not active():raise RuntimeError('User STOP/pause')
                    okay=False; gates={}
                    for host, suffix in [('openlibing-DevEnv-661495','h1'),('openlibing-DevEnv-055409','h2')]:
                        if suffix=='h2':
                            released=True
                            for github_status in ['in_progress','queued']:
                                observation=subprocess.run(['gh','run','list','--repo','yyttt6/TileOPs','--status',github_status,'--limit','30','--json','databaseId,status,workflowName'],capture_output=True,text=True)
                                if observation.returncode or json.loads(observation.stdout):
                                    released=False;gates['H2_task_release']={'github_status':github_status,'stdout':observation.stdout,'stderr':observation.stderr};break
                            if not released:continue
                        c.H=host;okay,gate=c.admit();gates[host]=gate
                        if okay:
                            label=f'rollout-golden42-nn-gemm7-{i}-golden42-ownes-runtime-{suffix}-v1'
                            save(host=host)
                            break
                    if okay:
                        save(status='guarded_full_native_API_15raw_new_golden',ordinal=i,label=label)
                        event=c.run('r-template','ops-nn-gemm10',[i],label)
                    else:
                        event=None;save(status='await_fresh_H1_queue_and_physical_release',ordinal=i,gate=gates)
                if event is None:
                    time.sleep(30)
                    continue
                assert 'batch' in event and len(event['batch']['results'])==1
                result=event['batch']['results'][0]
                assert result['status']=='raw_checked_pending_guard',json.dumps(result,default=str)
                assert result['reference_policy']=='audited_cpu_fp64_exact42_v1'
                base=D/'remote/runs'/event['tag']/str(i)
                correction=json.loads((base/'GOLDEN_CORRECTION.json').read_text())
                bound=next(r for r in cert['rows'] if r['ordinal']==i)
                assert correction['manifest_sha256']==sha(D/'GOLDEN42_CERTIFICATES.json')
                assert correction['certificate']==bound and correction['payload_sha256']==result['payload_sha256']
                assert correction['baseline_callable_kernel_precision_checker_tolerance_unchanged']
                save(status='audit_full_raw_source_ELF_and_golden_identity',ordinal=i)
                with (D.parents[2]/'.agent_memory/.maintenance.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    backup=D/'action-backups'/('golden42-accept-'+str(i)+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
                    backup.mkdir(parents=True,exist_ok=False)
                    for file in [D/'GOLDEN42_GEMM7_RAW_AUDIT.json',D/'RESOLVED_GOLDEN42_GEMM7.json']:
                        if file.exists():
                            raw=file.read_bytes();(backup/file.name).write_bytes(raw);(backup/(file.name+'.sha256')).write_text(hashlib.sha256(raw).hexdigest())
                    subprocess.run([sys.executable,str(D/'audit_golden42_gemm7_raw.py')],check=True)
                    out=summarize()
                    assert i in {r['ordinal'] for r in out['rows']}
                    subprocess.run([sys.executable,str(D/'effective_coverage.py')],check=True)
                save(status='qualified_native_case_next',last_qualified=i)
                break
        save(status='finite2_complete_next_unfinished_group_required')
    except BaseException as error:
        save(status='stopped_requires_diagnosis_no_numeric_retry',error=repr(error))
        raise
    finally:
        c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':record['status']})
