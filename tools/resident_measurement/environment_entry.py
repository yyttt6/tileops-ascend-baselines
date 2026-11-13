"""Guard foreground: verify isolated environment, then run unchanged measurement bytes."""
import argparse,fcntl,hashlib,json,pathlib,runpy,sys,traceback
from verify_env import verify
RUNNER_SHA='96f85c58ba0e0b0c8d9fd227a728f50d8f321607671fd7a43e7dde4bd4485560'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--receipt',type=pathlib.Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();package=pathlib.Path(__file__).resolve().parent
 cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 runner=pathlib.Path(cmd[0]).resolve();assert runner==package/'measure_driver_provenance_v2.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()==RUNNER_SHA
 record={'protocol':'round-measurement04-patched-alternate-pending','envID':'round-candidate03','status':'checking'}
 try:
  with open('/run/lock/ascend-npu/device-0.lock') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError('No held reservation')
  record['environment']=verify(a.root/'round-isolated-envs/round-candidate03',a.root/'venv','0be87418b543e882f4e7351a21a0a9bcf578b33a8be271c81b0e861d6da853db')
  record['status']='verified';a.receipt.write_text(json.dumps(record,indent=2)+'\n')
  sys.argv=cmd;runpy.run_path(str(runner),run_name='__main__')
 except BaseException:
  if record['status']!='verified':
   record['status']='failed';record['exception']=traceback.format_exc();a.receipt.write_text(json.dumps(record,indent=2)+'\n')
  raise
if __name__=='__main__':main()
