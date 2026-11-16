"""Sequential fail-fast handoff from actualH2build to qualifiedprivateLayerNorm5; no blind retry."""
from pathlib import Path
import json,subprocess,datetime,os,fcntl,time,hashlib,shlex,sys
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-layernorm-v1';H='openlibing-DevEnv-661495'
lock=(P/'layernorm-sum-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_originalSegment_device_build','queued_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts_as_primary_dispatch':False,'no_formal_coverage_until_originalchecker15raw_guard_source_acceptance':True}
def save(status):
 state.update(status=status,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p=P/'LAYERNORM_SUM_HANDOFF.json';t=p.with_suffix('.handoff.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
def stop():assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
def run(cmd):stop();subprocess.run(cmd,check=True)
try:
 save('CPU_wait_originalSegment_device_build')
 while True:
  stop();j=json.loads((A/'SERVER2_LAYERNORM_DEVICE_BUILD.json').read_text());s=j['status']
  if s=='built_pending_identity_and_load':assert j['exit_code']==0 and j['guard']['performance_valid'];break
  assert s not in ('reference_build_failed_requires_diagnosis','device_build_failed_requires_diagnosis','stopped_requires_diagnosis'),j
  time.sleep(10)
 save('CPU_assemble_sourceverified_original_package');run([sys.executable,'/tmp/assemble_native_layernorm_package.py'])
 save('CPU_deploy_original_dependency_allfilesSHA');run([sys.executable,'/tmp/deploy_layernorm_private_package.py'])
 p=P/'ops_nn_layernorm_api.py';dst=N+'/'+p.name;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');assert not p.exists();p.write_bytes(sys.stdin.buffer.read())';stop();subprocess.run(['ssh',H,'python3 -c '+shlex.quote(code)],input=p.read_bytes(),check=True);assert subprocess.check_output(['ssh',H,'sha256sum '+dst],text=True).split()[0]==hashlib.sha256(p.read_bytes()).hexdigest()
 save('guarded_actual_private_API_output_probe');run([sys.executable,'-u',str(P/'queue_server1_layernorm_attribution.py')])
 save('CPU_verify_actual_selected_load_launch_SHA');run([sys.executable,'/tmp/qualify_layernorm_private.py'])
 assert json.loads((A/'PRIVATE_LAYERNORM_QUALIFICATION.json').read_text())['private_original_device_kernels_verified']
 save('handoff_to_LayerNorm5_originalformal_controller')
 stop();os.execv(sys.executable,[sys.executable,'-u',str(P/'queue_ops_nn_layernorm.py')])
except BaseException as e:
 state['error']=repr(e);save('stopped_requires_diagnosis');raise
