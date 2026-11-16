from pathlib import Path
import subprocess,json,shlex,hashlib,datetime
D=Path('reports/baseline-over3-campaign-20261110/ordered100-20261112');A=D/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1';files={A/'probe_gemm_pad4.py':N+'/probe_math_gemm-pad4.py',D/'ops_nn_gemm_pad4_api.py':N+'/ops_nn_gemm_pad4_api.py',A/'gemm-pad4-probe-entry.sh':N+'/probe-entry.sh',A/'pad4-original-ascend_gemm_cases.py':N+'/original-ascend_gemm_cases.py',A/'pad4-original-ascend_nn_aligned_gemm.py':N+'/original-ascend_nn_aligned_gemm.py'}
# The actual executable filename is pinned by the entry, keep it a normal identifier.
files[A/'probe_gemm_pad4.py']=N+'/probe_gemm_pad4.py'
files[A/'gemm-pad4-runtime-env.sh']=N+'/runtime-env.sh'
proof={}
for host in ['openlibing-DevEnv-661495','openlibing-DevEnv-055409']:
 proof[host]={}
 for f,dest in files.items():
  raw=f.read_bytes();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');p.parent.mkdir(exist_ok=True);b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';r=subprocess.run(['ssh',host,'/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],input=raw,capture_output=True,check=True);h=hashlib.sha256(raw).hexdigest();assert r.stdout.decode().strip()==h;proof[host][dest]=h
 code='from pathlib import Path;import shutil,hashlib,json;n=Path('+repr(N)+');src=n.parent/"ops-math-dequant-v1/original-flush";dst=n/"original-flush";shutil.copytree(src,dst) if not dst.exists() else None;assert {str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob("*") if p.is_file()}=={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob("*") if p.is_file()};print(json.dumps({str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob("*") if p.is_file()}))';r=subprocess.run(['ssh',host,'/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],capture_output=True,check=True);proof[host]['original_flush']=json.loads(r.stdout)
(A/'GEMM_PAD4_PROBE_DEPLOYMENT.json').write_text(json.dumps(proof,indent=2)+'\n');print('Pad4 probes bothhosts staged exactSHA')
