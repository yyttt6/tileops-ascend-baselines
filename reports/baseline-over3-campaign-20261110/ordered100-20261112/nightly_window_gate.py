"""Read-only scheduled-nightly admission. 02:00 Asia/Shanghai, not UTC.

A passed result is only daily task evidence; fresh PR/worker/queue/physical
checks remain mandatory. No dispatch, cancellation or NPU operation here.
"""
import datetime,json,pathlib,subprocess,fcntl,os,email.utils
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
def decide(now,marker,runs,server_clock_offset_seconds=0.0):
 deadline,date=window(now);base={'scheduled_utc':deadline.isoformat(),'scheduled_timezone':'Asia/Shanghai','daily_time':'02:00','marker_date':date,'margin_seconds':750,'allowed':False}
 if now<deadline:return {**base,'reason':'scheduled_nightly_600plus150_margin'}
 if not marker or marker.get('status') not in ('sending','dispatched'):return {**base,'reason':'await_actual_dispatch_evidence_not_idle'}
 sent=parse(marker['at'])
 if sent<deadline-datetime.timedelta(seconds=60):return {**base,'reason':'wrong_scheduled_window_marker'}
 matches=[r for r in runs if r.get('event')=='workflow_dispatch' and r.get('head_branch')=='main' and parse(r['created_at'])>=(sent+datetime.timedelta(seconds=server_clock_offset_seconds)).replace(microsecond=0)]
 if not matches:return {**base,'reason':'dispatch_sent_no_actual_run_observed'}
 active=[r for r in matches if r['status']!='completed']
 return {**base,'allowed':not active,'reason':'actual_daily_runs_completed_require_fresh_physical_checks' if not active else 'actual_daily_nightly_task_active','matching_runs':matches,'marker':marker}
# Baseline admission consumes the authoritative one-night cancellation; it never
# changes the timer/service or treats elapsed clock time as a task release.
OFFICIAL_DATE_SKIPS={'2026-11-15':{
 'record':P.parents[2]/'reports/nightly-skip-20261115/RESULT.json',
 'record_sha256':'85d84954855ff745226cb0a015f5401c12c72e37be0230f78fa6bd29c19f97bb',
 'dropin':pathlib.Path('/etc/systemd/system/tileops-nightly-dispatch.service.d/skip-20261115.conf'),
 'dropin_sha256':'c44fa6635e006eceebf50ee798362a2bb5b02e5b692934811327d87ea36e5cf1',
 'condition':"sys.exit(datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat() == '2026-11-15')"}}
def official_skip(deadline,date):
 import hashlib
 spec=OFFICIAL_DATE_SKIPS.get(date)
 if spec is None:return None
 try:
  raw=spec['record'].read_bytes();drop=spec['dropin'].read_bytes()
  if hashlib.sha256(raw).hexdigest()!=spec['record_sha256'] or hashlib.sha256(drop).hexdigest()!=spec['dropin_sha256']:return None
  record=json.loads(raw)
  if parse(record['skip_bjt']).astimezone(UTC)!=deadline or parse(record['resume_bjt']).astimezone(UTC)!=deadline+datetime.timedelta(days=1):return None
  r=subprocess.run(['systemctl','show','tileops-nightly-dispatch.service','-p','DropInPaths','-p','ExecCondition','-p','Result','-p','ActiveState'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
  if r.returncode or str(spec['dropin']) not in r.stdout or spec['condition'] not in r.stdout:return None
  return {'scheduled_utc':deadline.isoformat(),'scheduled_timezone':'Asia/Shanghai','daily_time':'02:00','marker_date':date,'margin_seconds':750,'allowed':True,'reason':'official_one_night_skip_require_fresh_task_queue_guard','official_skip':{'record':str(spec['record']),'record_sha256':spec['record_sha256'],'dropin':str(spec['dropin']),'dropin_sha256':spec['dropin_sha256'],'loaded_systemd_condition_verified':True,'observed_service':r.stdout},'not_a_physical_idle_or_PR_release':True,'normal_schedule_unmodified':True}
 except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):return None

def check(now=None):
 fixed_now=now is not None;now=now or datetime.datetime.now(UTC);deadline,date=window(now)
 skipped=official_skip(deadline,date)
 if skipped is not None:return {**skipped,'checked_at':now.isoformat()}
 if now<deadline:return decide(now,None,[])
 try:marker=json.loads((MARKERS/(date+'.json')).read_text())
 except (OSError,json.JSONDecodeError):return decide(now,None,[])
 cache=P/'NIGHTLY_WINDOW_RELEASE_EVIDENCE.json';lock=P/'nightly-window-release.lock'
 with lock.open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  try:old=json.loads(cache.read_text())
  except (OSError,json.JSONDecodeError):old={}
  key=deadline.isoformat();prior=old.get(key,{})
  cmd=['gh','api','-i','-H','Cache-Control: no-cache',f'repos/{REPO}/actions/workflows/ascend950-nightly.yml/runs?per_page=100']
  query_started=datetime.datetime.now(UTC)
  try:r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  except subprocess.TimeoutExpired:return {'allowed':False,'reason':'nightly_history_query_timeout','scheduled_utc':key}
  if r.returncode:return {'allowed':False,'reason':'nightly_history_query_failed','scheduled_utc':key,'query_failed':True,'error':r.stderr}
  query_finished=datetime.datetime.now(UTC);headers,sep,body=r.stdout.partition('\n\n');date_lines=[line.split(':',1)[1].strip() for line in headers.splitlines() if line.lower().startswith('date:')];assert sep and len(date_lines)==1,'FreshGitHubHTTPDate required';server_time=email.utils.parsedate_to_datetime(date_lines[0]);midpoint=query_started+(query_finished-query_started)/2;offset=(server_time-midpoint).total_seconds()
  if abs(offset)>300:return {'allowed':False,'reason':'server_clock_offset_exceeds_verified_bound','offset_seconds':offset,'scheduled_utc':key}
  rows=json.loads(body)['workflow_runs'];known={x['id'] for x in prior.get('matching_runs',[])};present={x['id'] for x in rows}
  # A long run must not vanish behind newer manual runs/historylimit.
  for rid in known-present:
   r=subprocess.run(['gh','api',f'repos/{REPO}/actions/runs/{rid}'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
   if r.returncode:return {'allowed':False,'reason':'known_daily_run_query_failed','scheduled_utc':key,'query_failed':True,'run_id':rid,'error':r.stderr}
   rows.append(json.loads(r.stdout))
  filtered=[{k:x[k] for k in ['id','status','conclusion','event','head_branch','created_at']} for x in rows]
  final_now=now if fixed_now else datetime.datetime.now(UTC);result=decide(final_now,marker,filtered,offset);result['checked_at']=final_now.isoformat();result['clock_calibration']={'server_HTTP_Date':server_time.isoformat(),'local_query_started':query_started.isoformat(),'local_query_finished':query_finished.isoformat(),'server_minus_local_seconds':offset,'roundtrip_seconds':(query_finished-query_started).total_seconds()};old[result['scheduled_utc']]=result;tmp=cache.with_suffix('.nightly.tmp');tmp.write_text(json.dumps(old,indent=2)+'\n');os.replace(tmp,cache);return result
