from pathlib import Path
import subprocess,hashlib,json,shlex,datetime
D=Path('reports/baseline-over3-campaign-20261110/ordered100-20261112');A=D/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-poolmean-v1';m=json.loads((A/'POOLMEAN_NATIVE_PACKAGE.server2.json').read_text());root=m['staging_root'];arc=A/'POOLMEAN_ORIGINAL_HOST_PACKAGE.tar.gz';assert not arc.exists()
with arc.open('wb') as f:subprocess.run(['ssh','openlibing-DevEnv-055409','tar -czf - -C '+shlex.quote(N)+' complete-original-api-host-v1'],stdout=f,check=True)
# immutable taskprivate dependency install, sourceH2 retained; no experiment inputs orcold data moved.
check='test ! -e '+shlex.quote(root)+' && tar -xzf - -C '+shlex.quote(N)
with arc.open('rb') as f:subprocess.run(['ssh','openlibing-DevEnv-661495',check],stdin=f,check=True)
p=A/'POOLMEAN_NATIVE_PACKAGE.server2.json';dst=N+'/POOLMEAN_NATIVE_PACKAGE.json';code='from pathlib import Path;import sys;p=Path('+repr(dst)+');assert not p.exists();p.write_bytes(sys.stdin.buffer.read())';subprocess.run(['ssh','openlibing-DevEnv-661495','python3 -c '+shlex.quote(code)],input=p.read_bytes(),check=True)
code='''import pathlib,json,hashlib
n=pathlib.Path(NPATH);m=json.loads((n/'POOLMEAN_NATIVE_PACKAGE.json').read_text());r=pathlib.Path(m['staging_root'])
for k,v in m['all_files'].items():assert hashlib.sha256((r/k).read_bytes()).hexdigest()==v['sha256'],k
print(json.dumps({'files':len(m['all_files']),'bytes':sum(v['bytes'] for v in m['all_files'].values()),'status':'all_original_host_API_package_files_identical_H1_H2_no_private_device_claim'}))
'''.replace('NPATH',repr(N));r=subprocess.check_output(['ssh','openlibing-DevEnv-661495','python3 -c '+shlex.quote(code)],text=True);proof=json.loads(r);proof.update(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=hashlib.sha256(arc.read_bytes()).hexdigest(),source_H2_retained=True,no_cold_inputs_migrated=True,not_H1_runtime_acceptance=True);(A/'POOLMEAN_HOST_PACKAGE_DEPLOYMENT_H1.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof)
