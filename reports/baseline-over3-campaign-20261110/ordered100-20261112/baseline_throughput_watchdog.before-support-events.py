#!/usr/bin/env python3
"""CPU-only liveness/rolling-throughput observer; never submits device work."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
P=ROOT/'reports/baseline-over3-campaign-20261110/ordered100-20261112'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def write(p,x):
 t=p.with_name(p.name+'.watchdog.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)

def active_controllers():
 result=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   args=proc.joinpath('cmdline').read_bytes().decode().split('\0');cwd=proc.joinpath('cwd').resolve()
   for arg in args[1:]:
    path=pathlib.Path(arg)
    if not path.name.startswith(('queue_','expand_','continue_ordered')) or path.suffix!='.py':continue
    resolved=(path if path.is_absolute() else cwd/path).resolve()
    if resolved.parent==P:
     result.append({'pid':int(proc.name),'entry':path.name,'start_ticks':proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]});break
  except (OSError,ValueError,UnicodeError):pass
 return result

def primary_dispatch_controllers(controllers):
 roles=read(P/'CONTROLLER_ROLES.json',{})
 return [c for c in controllers if roles.get(c['entry'],{}).get('counts_as_primary_dispatch',True)]

def paired_results(now):
 records={}
 for pattern in ['SGL_*CURRENT.json','VLLM_*CURRENT.json','FLA_*CURRENT.json']:
  for f in P.glob(pattern):
   obj=read(f,{})
   for row in obj.get('rows',[]):
    if row.get('guard_checker_15raw_valid') is not True:continue
    old=row.get('original_checkpoint');new=row.get('candidate_checkpoint')
    if not old or not new or not (P/old).is_file() or not (P/new).is_file():continue
    identity=(old,new);records[identity]={'ordinal':row.get('ordinal'),'completed_at_local':max((P/old).stat().st_mtime,(P/new).stat().st_mtime),'summary':f.name}
 return records

def alarm_due(now,reason):
 schedule=P/'RECURRING_ALARMS.json'
 with schedule.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);s=read(schedule,{})
  if s.get('disabled') or not s.get('interval_seconds'):return 'disabled_no_change'
  if s.get('pending'):return 'delivery_already_pending'
  if any(m['due_at']<=now and 'delivered_at' not in m for m in s.get('milestones',[])):return 'overdue_reminder_already_armed'
  s.setdefault('milestones',[]).append({'due_at':now,'after_seconds':now-s['start_at'],'watchdog_reason':reason})
  write(schedule,s);return 'immediate_inspection_armed'

def main():
 now=time.time();out=P/'THROUGHPUT_WATCHDOG.json'
 with (P/'throughput-watchdog.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=read(out,{})
  if (P/'STOP').exists() or (P.parent/'STOP').exists() or read(P.parent/'STATE.json',{}).get('status')!='active':
   write(out,{'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paused':True,'action':'No reminder or work dispatch; user pause/stop wins'});return
  controllers=active_controllers();primary=primary_dispatch_controllers(controllers);pairs=paired_results(now);recent=[x for x in pairs.values() if now-3600<=x['completed_at_local']<=now]
  coverage=read(P/'CURRENT_VALID_COVERAGE.json',{}).get('effective_coverage');samples=[x for x in previous.get('coverage_observations',[]) if x['at']>=now-7200];samples.append({'at':now,'coverage':coverage})
  anchor=next((x for x in reversed(samples) if x['at']<=now-3600),None)
  growth=coverage-anchor['coverage'] if anchor and coverage is not None else None
  idle_since=None if primary else previous.get('idle_since',now)
  current=read(P/'CURRENT_PROCESS.json',{})
  dispatch_event=None
  if not primary and current.get('status') in ('group_finished_next_dispatch_required','completed_group_no_active_controller'):
   dispatch_event={'kind':'group_finished_needs_successor','entry':current.get('entry'),'pid':current.get('prior_pid'),'at':current.get('checked_at'),'outcome':current.get('outcome')}
  elif not primary and current.get('status')=='active' and current.get('pid'):
   dispatch_event={'kind':'recorded_active_controller_missing','entry':current.get('entry'),'pid':current.get('pid'),'at':current.get('updated_at')}
  urgent=dispatch_event is not None and dispatch_event!=previous.get('last_handled_dispatch_event')
  reasons=[]
  if urgent:reasons.append(dispatch_event['kind'])
  if idle_since is not None and now-idle_since>=300:reasons.append('pending_scope_without_live_controller_for_5min')
  if len(recent)==0:reasons.append('zero_audited_API_pairs_last_hour')
  report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checked_at_epoch':now,'active_controllers':controllers,'primary_dispatch_controllers':primary,'idle_since':idle_since,'effective_coverage':coverage,'coverage_observations':samples,'observed_new_coverage_last_hour':growth,'coverage_growth_unknown_until_full_observation_window':anchor is None,'audited_API_pairs_last_hour':len(recent),'audited_API_pair_evidence':recent,'counting':'Only source summaries with full guard/checker15raw qualification and complete checkpoint pairs; no attempts/rejections/traversal counted. New coverage separately observed.','issues':reasons,'last_alert_at':previous.get('last_alert_at'),'device_work_submitted':False}
  report['last_handled_dispatch_event']=dispatch_event if urgent else previous.get('last_handled_dispatch_event')
  report['dispatch_event_requires_immediate_inspection']=urgent
  if reasons and (urgent or now-(previous.get('last_alert_at') or 0)>=1800):
   report['action']=alarm_due(now,','.join(reasons));report['last_alert_at']=now
  elif reasons:report['action']='Existing inspection/remediation cooldown; real process and throughput remain checked each minute'
  else:report['action']='Continue observing; no anomaly detected in monitored dimensions'
  write(out,report)
  with (P/'THROUGHPUT_WATCHDOG_HISTORY.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in report.items() if k not in ['coverage_observations','audited_API_pair_evidence']})+'\n')
  print(json.dumps({k:report[k] for k in ['checked_at','audited_API_pairs_last_hour','active_controllers','issues','action']}))
if __name__=='__main__':main()
