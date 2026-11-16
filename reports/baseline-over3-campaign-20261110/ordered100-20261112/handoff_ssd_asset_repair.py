"""Foreground safe-boundary handoff; no detached child and no device overlap."""
import pathlib,os,time,json,runpy,datetime,sys
P=pathlib.Path(__file__).resolve().parent
handoff=P/'SSD_ASSET_HANDOFF.json'
def put(status,**extra):
 handoff.write_text(json.dumps({'status':status,'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2))
put('waiting_previous_controller_safe_boundary',previous_pid=914840)
for _ in range(900):
 if not pathlib.Path('/proc/914840').exists():break
 time.sleep(1)
else:
 put('handoff_timeout_previous_controller_not_interrupted');raise RuntimeError('No safe handoff; inspect old process, no duplicate controller started')
runpy.run_path(str(P/'resume_ssd_asset_repair.py'),run_name='__main__')
put('resuming_ordered_controller_after_requeue')
f=P/'CURRENT_PROCESS.json';old=f.read_bytes();f.with_name(f.name+'.before-ssdasset-handoff').write_bytes(old);j=json.loads(old);j.update(pid=os.getpid(),foreground_tool_session='see SSD_ASSET_HANDOFF_TOOL.json',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),safe_handoff='SSD_ASSET_HANDOFF.json');f.write_text(json.dumps(j,indent=2))
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
