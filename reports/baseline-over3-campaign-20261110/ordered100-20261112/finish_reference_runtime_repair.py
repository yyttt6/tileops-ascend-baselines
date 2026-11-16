"""Task-private path-only rebase, full identity gate, then freshguard RMS pilot."""
import pathlib,json,subprocess,hashlib,time,datetime,sys,fcntl
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
N=P/'reference-runtime-launcher-v1';R=c.R+'/reference-runtime-torch210-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2))
proof=P/'REFERENCE_RUNTIME_PRIVATE_COPY.json'
# The foreground copy must finish successfully before any deployment/guard launch.
while not proof.exists():
 if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('user STOP')
 alive=subprocess.run(['pgrep','-f','^python3 -u /tmp/repair_baseline_runtime.py$'],stdout=subprocess.DEVNULL).returncode==0
 if not alive:raise RuntimeError('copy exited without verified manifest')
 time.sleep(5)
j=json.loads(proof.read_text());assert j['root']==R and all(x['all_equal'] for x in j['records'])
for name in ['private_python.py','allpairs-python','vendor_python.sh','vendor_bootstrap.py','verify_runtime.py','ORIGINAL_RUNTIME_IDENTITY.json','PATH_REBASE.json']:
 data=(N/name).read_bytes();code='import pathlib,sys;p=pathlib.Path('+repr(R+'/'+name)+');p.write_bytes(sys.stdin.buffer.read())'
 subprocess.run(['ssh',c.H,'python3 -c '+__import__('shlex').quote(code)],input=data,check=True)
 actual=c.remote('sha256sum '+R+'/'+name,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==sha(N/name)
c.remote('chmod +x '+R+'/allpairs-python',check=True)
g=c.remote(c.PY+R+'/verify_runtime.py',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
put(P/'REFERENCE_RUNTIME_REBASED_IDENTITY_GATE.json',{'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':g.returncode,'output':g.stdout,'original_expected_hashes_unchanged':True,'tree_proof':proof.name,'launchers':{p.name:sha(p) for p in N.iterdir() if p.is_file()},'NPU_imported':False})
assert g.returncode==0,g.stdout
# Only this task's inactive launch file changes; original shared launchers untouched.
f=P/'launch.sh';raw=f.read_bytes();assert b'/opt/baseline-over3-20261110/vendor-runtime-repair/vendor_python.sh' in raw
backup=P/'launch.before-private-reference-runtime-20261113T0650.sh';assert not backup.exists();backup.write_bytes(raw)
s=raw.decode();old='exec bash /opt/baseline-over3-20261110/vendor-runtime-repair/vendor_python.sh "$profile"';s=s.replace(old,'exec bash '+R+'/vendor_python.sh "$profile"')
old='exec /opt/tileops-all-pairs/bin/allpairs-python "$P/rollout_bootstrap.py" "$@"';assert s.count(old)==1;s=s.replace(old,'exec '+R+'/allpairs-python "$P/rollout_bootstrap.py" "$@"')
f.write_text(s);data=f.read_bytes();code='import pathlib,sys;p=pathlib.Path('+repr(c.R+'/launch.sh')+');old=p.read_bytes();backup=p.with_name("launch.before-private-reference-runtime-20261113T0650.sh");assert not backup.exists();backup.write_bytes(old);p.write_bytes(sys.stdin.buffer.read())'
subprocess.run(['ssh',c.H,'python3 -c '+__import__('shlex').quote(code)],input=data,check=True);assert c.remote('sha256sum '+c.R+'/launch.sh',stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==sha(f)
put(P/'REFERENCE_RUNTIME_LAUNCH_DEPLOYMENT.json',{'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':sha(backup),'after':sha(f),'shared_runtime_changed':False,'scope':'task-private r-sgl/r-vllm reference launcher paths only; native providers still require their own allowed-root checks; no full-native reacceptance claim','identity_gate':'REFERENCE_RUNTIME_REBASED_IDENTITY_GATE.json'})
# New label retains original v1 precompute failure unchanged. CPU prepare then c.run freshgate.
label='rollout-vllm-rms-api-candidate-1214-v2-private-reference'
with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-vllm prepare '+label+'-prepare 1214',stdout=out,stderr=subprocess.STDOUT).returncode
assert rc==0,'CPU prepare failed, do not device retry'
while True:
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-vllm','vllm-rms-api',[1214],label)
 if e is not None:break
 time.sleep(30)
assert 'batch' in e,'new guarded attempt failed; retain evidence/no blind retry'
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
put(P/'VLLM_RMS_API_PILOT_RESULT.json',{'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':label+'.CHECKPOINT.json','result':e['batch']['results'][0],'original_control':'ROLLOUT_LEDGER.json:1214','selection_pending_control_fidelity_proof':True,'repaired_runtime':'REFERENCE_RUNTIME_REBASED_IDENTITY_GATE.json','no_duplicate_effective_coverage':True})
