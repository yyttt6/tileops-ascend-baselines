"""Sequential fail-fast handoff from actualH2build to qualifiedprivateSum7; no blind retry."""
from pathlib import Path
import json,subprocess,datetime,os,fcntl,time,hashlib,shlex,sys
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1';H='openlibing-DevEnv-661495'
lock=(P/'segment-sum-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_originalSegment_device_build','queued_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts_as_primary_dispatch':False,'no_formal_coverage_until_originalchecker15raw_guard_source_acceptance':True}
def save(status):
 state.update(status=status,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p=P/'SEGMENT_SUM_HANDOFF.json';t=p.with_suffix('.handoff.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
def stop():assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
def run(cmd):stop();subprocess.run(cmd,check=True)
try:
 save('CPU_resume_after_packagev3_SHA_verified_deployment')
 j=json.loads((A/'SERVER2_SEGMENT_DEVICE_BUILD_V4.json').read_text());assert j['status']=='built_pending_identity_and_load' and j['exit_code']==0 and j['guard']['performance_valid']
 m=json.loads((A/'SEGMENT_PRIVATE_PACKAGE_DEPLOYMENT_H1.json').read_text());assert m['status']=='all_original_host_device_package_files_identical_H1_H2'
 p=P/'ops_nn_segment_api.py';dst=N+'/'+p.name;assert subprocess.check_output(['ssh',H,'sha256sum '+dst],text=True).split()[0]==hashlib.sha256(p.read_bytes()).hexdigest()
 # Immutable exact existingbinding is reused, not rewritten; packagev3 alreadySHAverified.
 save('guarded_actual_private_API_output_probe');run([sys.executable,'-u',str(P/'queue_server1_segment_attribution_v2.py')])
 save('CPU_verify_actual_selected_load_launch_SHA');run([sys.executable,'/tmp/qualify_segment_private_v3.py'])
 assert json.loads((A/'PRIVATE_SEGMENT_QUALIFICATION.json').read_text())['private_original_device_kernels_verified']
 save('handoff_to_Sum7_originalformal_controller')
 stop();os.execv(sys.executable,[sys.executable,'-u',str(P/'queue_ops_nn_segment.py')])
except BaseException as e:
 state['error']=repr(e);save('stopped_requires_diagnosis');raise
