"""Read-only scheduled-nightly admission. 02:00 Asia/Shanghai, not UTC.

A passed result is only daily task evidence; fresh PR/worker/queue/physical
checks remain mandatory. No dispatch, cancellation or NPU operation here.
"""
import datetime,json,pathlib,subprocess,fcntl,os
from zoneinfo import ZoneInfo
P=pathlib.Path(__file__).resolve().parent
UTC=datetime.timezone.utc
SH=ZoneInfo('Asia/Shanghai')
MARKERS=pathlib.Path('/var/lib/tileops-nightly-dispatch')
REPO='yyttt6/TileOPs.github.io'
def window(now):
 if now.tzinfo is None:raise ValueError('Explicit UTC/aware time required')
 deadline=now.astimezone(SH).replace(hour=2,minute=0,second=0,microsecond=0).astimezone(UTC)
 if now<deadline-datetime.timedelta(seconds=750):deadline-=datetime.timedelta(days=1)
 return deadline,deadline.astimezone(SH).date().isoformat()
def parse(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def decide(now,marker,runs):
 deadline,date=window(now);base={'scheduled_utc':deadline.isoformat(),'scheduled_timezone':'Asia/Shanghai','daily_time':'02:00','marker_date':date,'margin_seconds':750,'allowed':False}
 if now<deadline:return {**base,'reason':'scheduled_nightly_600plus150_margin'}
 if not marker or marker.get('status') not in ('sending','dispatched'):return {**base,'reason':'await_actual_dispatch_evidence_not_idle'}
 sent=parse(marker['at'])
 if sent<deadline-datetime.timedelta(seconds=60):return {**base,'reason':'wrong_scheduled_window_marker'}
 matches=[r for r in runs if r.get('event')=='workflow_dispatch' and r.get('head_branch')=='main' and parse(r['created_at'])>=sent.replace(microsecond=0)]
 if not matches:return {**base,'reason':'dispatch_sent_no_actual_run_observed'}
 active=[r for r in matches if r['status']!='completed']
 return {**base,'allowed':not active,'reason':'actual_daily_runs_completed_require_fresh_physical_checks' if not active else 'actual_daily_nightly_task_active','matching_runs':matches,'marker':marker}
def check(now=None):
 fixed_now=now is not None;now=now or datetime.datetime.now(UTC);deadline,date=window(now)
 if now<deadline:return decide(now,None,[])
 try:marker=json.loads((MARKERS/(date+'.json')).read_text())
 except (OSError,json.JSONDecodeError):return decide(now,None,[])
 cache=P/'NIGHTLY_WINDOW_RELEASE_EVIDENCE.json';lock=P/'nightly-window-release.lock'
 with lock.open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  try:old=json.loads(cache.read_text())
  except (OSError,json.JSONDecodeError):old={}
  key=deadline.isoformat();prior=old.get(key,{})
  cmd=['gh','api',f'repos/{REPO}/actions/workflows/ascend950-nightly.yml/runs?per_page=100']
  try:r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  except subprocess.TimeoutExpired:return {'allowed':False,'reason':'nightly_history_query_timeout','scheduled_utc':key}
  if r.returncode:return {'allowed':False,'reason':'nightly_history_query_failed','scheduled_utc':key,'query_failed':True,'error':r.stderr}
  rows=json.loads(r.stdout)['workflow_runs'];known={x['id'] for x in prior.get('matching_runs',[])};present={x['id'] for x in rows}
  # A long run must not vanish behind newer manual runs/historylimit.
  for rid in known-present:
   r=subprocess.run(['gh','api',f'repos/{REPO}/actions/runs/{rid}'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
   if r.returncode:return {'allowed':False,'reason':'known_daily_run_query_failed','scheduled_utc':key,'query_failed':True,'run_id':rid,'error':r.stderr}
   rows.append(json.loads(r.stdout))
  filtered=[{k:x[k] for k in ['id','status','conclusion','event','head_branch','created_at']} for x in rows]
  final_now=now if fixed_now else datetime.datetime.now(UTC);result=decide(final_now,marker,filtered);result['checked_at']=final_now.isoformat();old[result['scheduled_utc']]=result;tmp=cache.with_suffix('.nightly.tmp');tmp.write_text(json.dumps(old,indent=2)+'\n');os.replace(tmp,cache);return result
