"""CPU-only real inspection. Never queries the NPU driver or submits device work.

Controllers own CURRENT_PROCESS and business decisions; this inspector updates only
inspection/counter fields, preserving newer dispatch and strategy facts.
"""
from pathlib import Path
import argparse,ast,concurrent.futures,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,shlex,time
D=Path(__file__).resolve().parent;P=D.parent;ROOT=D.parents[2]
def utc():return datetime.datetime.now(datetime.timezone.utc)
def digest(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text())
def atomic(p,value):
 raw=value if isinstance(value,bytes) else (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();t=p.with_suffix('.inspection.tmp');t.write_bytes(raw);os.replace(t,p)
def event_logs():
 p=D/'QUALIFIED_EVENT_LOGS.json'
 if p.exists():return read(p)['logs']
 tree=ast.parse(Path('/tmp/inspect_baseline_current.py').read_text());logs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='logs' for t in n.targets))
 for name in ['ops-nn-where-queued.log','ops-nn-maskedfill-queued.log']:
  if name not in logs:logs.append(name)
 assert all(Path(x).name==x for x in logs)
 with (ROOT/'.agent_memory/.maintenance.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not p.exists():atomic(p,{'logs':logs,'counting':'Only qualified effective_coverage snapshots, originalscope2119. Attempts/rejections/compile/probes excluded.'})
 return read(p)['logs']
def growth(cov,now):
 cut=datetime.datetime.fromisoformat(cov['checked_at']);events=[]
 for name in event_logs():
  p=D/name
  if not p.exists():continue
  for line in p.read_text(errors='replace').splitlines():
   if not line.startswith('{') or 'effective_coverage' not in line:continue
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('scope_total')!=2119:continue
   at=datetime.datetime.fromisoformat(r['checked_at'])
   if at<=cut:events.append((at,r['effective_coverage']))
 events.sort();assert events and events[-1][1]==cov['effective_coverage'],'Coverage authority/event stream disagreement; do not invent growth'
 def delta(seconds):
  anchors=[v for t,v in events if t<=now-datetime.timedelta(seconds=seconds)]
  return cov['effective_coverage']-anchors[-1] if anchors else None
 return delta(3600),delta(1800)
REMOTE=r'''import json,pathlib,os,shutil,subprocess,datetime
p=pathlib.Path('/run/lock/ascend-npu/device-0.lock');g={};errors=[]
try:g=json.loads(p.read_text())
except Exception as e:errors.append('guard_metadata:'+type(e).__name__)
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def proc(row):
 try:
  q=pathlib.Path('/proc')/str(row['pid']);stat=q.joinpath('stat').read_text().rsplit(')',1)[1].split();args=q.joinpath('cmdline').read_bytes().decode().split(chr(0));return {'pid':row['pid'],'start_ticks':int(stat[19]),'state':stat[0],'matches_recorded_start':int(stat[19])==row.get('start_ticks'),'comm':q.joinpath('comm').read_text().strip(),'argv_names':[pathlib.Path(a).name for a in args[:2]]}
 except Exception:return {'pid':row.get('pid'),'exists':False}
entries=[]
try:
 st=p.stat()
 for line in pathlib.Path('/proc/locks').read_text().splitlines():
  a=line.split()
  if 'FLOCK' not in a or '->' in a:continue
  for i,x in enumerate(a):
   bits=x.split(':')
   if len(bits)!=3:continue
   try:match=int(bits[0],16)==os.major(st.st_dev) and int(bits[1],16)==os.minor(st.st_dev) and int(bits[2])==st.st_ino
   except ValueError:continue
   if match:entries.append({'pid':int(a[i-1]),'mode':a[i-2],'lock_type':'FLOCK'})
except Exception as e:errors.append('proc_locks:'+type(e).__name__)
workers=[]
for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
 a=line.split()
 if len(a)>1 and a[1].endswith('Runner.Worker'):workers.append({'pid':int(a[0]),'program':pathlib.Path(a[1]).name})
print(json.dumps({'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'free_bytes':shutil.disk_usage('/').free,'queue':[x.name for x in pathlib.Path('/run/lock/ascend-npu/device-0.queue').iterdir()],'workers':workers,'guard_metadata':g,'guard_boot_matches':g.get('boot_id')==boot,'guard_process':proc(g) if g.get('pid') else None,'child_process':proc(g['child']) if g.get('child') else None,'visible_kernel_flocks':entries,'errors':errors,'observation':'Read-only /proc, kernel flock table and guard-owned observations. No second driver probe; fresh submission physical checks unchanged.'}))
'''
def resource(host):
 try:
  r=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',host,'/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(REMOTE)],capture_output=True,text=True,timeout=20)
  if r.returncode==0:return host,json.loads(r.stdout)
  error={'returncode':r.returncode,'stderr':r.stderr[-1500:]}
 except (subprocess.TimeoutExpired,ValueError) as e:error={'exception':type(e).__name__,'detail':str(e)[-1000:]}
 return host,{'observed_at':utc().isoformat(),'reachable':False,'free_bytes':None,'queue':None,'workers':None,'errors':[error],'observation':'SSH observation unavailable; no idle/disk/release inferred. Other host inspection continues; no device query or submission.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--alarm-id',required=True);ap.add_argument('--due',required=True);ap.add_argument('--actions',required=True);ap.add_argument('--support-state');a=ap.parse_args();started=utc()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:resources=dict(pool.map(resource,['openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
 now=utc();cov=read(D/'CURRENT_VALID_COVERAGE.json');last60,last30=growth(cov,now)
 spec=importlib.util.spec_from_file_location('baseline_watchdog',ROOT/'tools/baseline_throughput_watchdog.py');wd=importlib.util.module_from_spec(spec);spec.loader.exec_module(wd);controllers=wd.active_controllers();primary=wd.primary_dispatch_controllers(controllers);cur=read(D/'CURRENT_PROCESS.json');pairs=[v for v in wd.paired_results(now.timestamp()).values() if v['completed_at_local']>=now.timestamp()-3600]
 from qualified_final_closures import collect
 finals=collect(D,now)
 report={'checked_at':now.isoformat(),'inspection_started_at':started.isoformat(),'alarm_id':a.alarm_id,'original_due_at':a.due,'effective_coverage':cov['effective_coverage'],'scope_total':2119,'new_qualified_coverage_last60min':last60,'new_qualified_coverage_last30min':last30,'new_complete_original_candidate_pairs_last60min':len(pairs),'new_final_under3_closures_last60min':sum(x['under3'] for x in finals),'new_final_closures_last60min':len(finals),'final_closure_evidence':finals,'counting':'Qualified measurement coverage, complete audited pairs, final ratio closures separately. Final closures count only exact current-runtime/protocol/input pairs with checkpointSHA/guard and generated identity audit. Compile/probe/attempt/refusal/traversal excluded.','actual_current_process':cur,'actual_controllers':controllers,'actual_primary_dispatch':primary,'recorded_primary_is_live':any(c['pid']==cur.get('pid') and c['entry']==cur.get('entry') for c in primary),'resources':resources,'actions':read(Path(a.actions)),'snapshot_read_only':True,'no_device_work_submitted_by_inspection':True}
 if a.support_state:
  f=D/a.support_state;v=read(f);report['support']={'record':a.support_state,'sha256':digest(f.read_bytes()),'state':{k:v.get(k) for k in ['entry','pid','host','queued_at','status','tag','checked_at','exit_code','error']}}
 name='INSPECTION_'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json';back=D/'memory-history'/now.strftime('%Y%m%dT%H%M%S%fZ')
 with (ROOT/'.agent_memory/.maintenance.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);back.mkdir(parents=True)
  def save(p,value):
   old=p.read_bytes() if p.exists() else b'';(back/p.name).write_bytes(old);(back/(p.name+'.sha256')).write_text(digest(old));assert (p.read_bytes() if p.exists() else b'')==old;atomic(p,value)
  save(D/name,report);s=read(P/'STATE.json');s.update(updated_at=now.isoformat(),last_inspection='ordered100-20261112/'+name);s['continuation'].update(current_effective_coverage=cov['effective_coverage'],current_unresolved_recorded=2119-cov['effective_coverage']);save(P/'STATE.json',s)
  lines=(P/'CURRENT.md').read_text().splitlines()
  for i,line in enumerate(lines):
   if line.startswith('- **有效覆盖'):lines[i]=f'- **有效覆盖{cov["effective_coverage"]}/2119（{cov["percentage"]:.2f}%），{2119-cov["effective_coverage"]}待闭环**；[实时计数](ordered100-20261112/CURRENT_VALID_COVERAGE.json)。覆盖、原新对照、最终<3及发布分开。'
   if line.startswith('- [实际控制器]'):lines[i]=f'- [实际控制器](ordered100-20261112/CURRENT_PROCESS.json)配合/proc，最新巡检[{name}](ordered100-20261112/{name})；CPU构建/证明/拒绝不计覆盖。'
  save(P/'CURRENT.md',('\n'.join(lines)+'\n').encode())
 p=D/'RECURRING_ALARMS.json'
 with p.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old=p.read_bytes();s=json.loads(old);(back/p.name).write_bytes(old);(back/(p.name+'.sha256')).write_text(digest(old));ack={'original_due_at':a.due,'checked_at':now.isoformat(),'inspection_receipt':str(D/name),'actions':report['actions']};matched=0
  for m in s.get('milestones',[]):
   if m.get('alarm_id')==a.alarm_id:m.update(inspection_checked_at=now.timestamp(),inspection_receipt=str((D/name).resolve()),inspection_action='; '.join(report['actions']));matched+=1
  s.setdefault('delivered_alarm_inspections',{})[a.alarm_id]=ack;s.update(last_inspection_at=now.timestamp(),last_inspection_receipt=str((D/name).resolve()));assert p.read_bytes()==old;atomic(p,s)
 print(json.dumps({'receipt':name,'alarm_matches':matched,'coverage':cov['effective_coverage'],'percentage':cov['percentage'],'last60':last60,'last30':last30,'pairs60':len(pairs),'primary_live':report['recorded_primary_is_live'],'root_free_GB':{h:round(v['free_bytes']/1e9,1) if v.get('free_bytes') is not None else None for h,v in resources.items()}},ensure_ascii=False))
if __name__=='__main__':main()
