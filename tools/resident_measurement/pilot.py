"""Two-case review-only adapter: unchanged measurement runner, resident04 sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os
from integration import run_candidate

PROTOCOL='round-measurement04-patched-alternate-pending'
RUNNER_SHA='96f85c58ba0e0b0c8d9fd227a728f50d8f321607671fd7a43e7dde4bd4485560'


def command(root,package,index,output):
    if index not in range(4):raise ValueError('only four reviewed Round contracts')
    return [str(root/'round-isolated-envs/round-candidate03/bin/python3'),str(package/'environment_entry.py'),
            '--root',str(root),'--receipt',str(output/'environment.json'),'--',str(package/'measure_driver_provenance_v2.py'),
            '--root',str(root),'--claim',str(package/'CLAIM.json'),
            '--index',str(index),'--windows','15','--output',str(output/'measurement')]


def verify(package,root,manifest_sha):
    raw=(package/'MANIFEST.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest_sha:raise ValueError('manifest mismatch')
    for name,h in json.loads(raw)['files'].items():
        path=(package/name).resolve()
        if not path.is_relative_to(package.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError(name)
    if hashlib.sha256((package/'measure_driver_provenance_v2.py').read_bytes()).hexdigest()!=RUNNER_SHA:raise ValueError('runner changed')
    for name,h in json.loads((package/'source-pins.json').read_text()).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('frozen source changed: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,choices=range(4),required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args()
    package=Path(__file__).resolve().parent
    verify(package,a.root,a.manifest_sha);a.output.mkdir(parents=True,exist_ok=False)
    row=json.loads((package/'CLAIM.json').read_text())['rows'][a.index]
    (a.output/'protocol.json').write_text(json.dumps({'effective_protocol':PROTOCOL,'status':'alternate/pending','contract':row,'limits':{'wait':600,'foreground':600,'sample_deadline':10,'request_cap':1024,'audit_bytes':8*1024*1024},'runner_sha256':RUNNER_SHA},indent=2))
    env=a.root/'round-isolated-envs/round-candidate03'
    os.environ.update(VIRTUAL_ENV=str(env),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    os.environ['PATH']=str(env/'bin')+os.pathsep+os.environ['PATH']
    for key,name in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('ASCEND_CACHE_PATH','ascend'),('TILELANG_CACHE_DIR','tilelang'),('TRITON_CACHE_DIR','triton')]:os.environ[key]=str(a.output/'cache'/name)
    args=argparse.Namespace(command=command(a.root,package,a.index,a.output),owner=PROTOCOL,wait=600,poll=1,idle_seconds=3,timeout=600)
    with (a.output/'worker.log').open('w') as log:
        code=run_candidate(args,[sys.executable,str(package/'real_worker.py')],log,a.output/'receipt.json',arm='A',max_requests=1024)
    # Retained field provenance in original runner is not effective protocol.
    verify(package,a.root,a.manifest_sha)
    return code
if __name__=='__main__':raise SystemExit(main())
