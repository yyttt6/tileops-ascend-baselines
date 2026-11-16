import pathlib,sys,json,subprocess,shlex,fcntl,time,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'math-transformer-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
results=[]
for i in [212,736]:
 label='rollout-registry-recovery-pilot-'+str(i)+'-v1'
 code='''import pathlib,json,hashlib
p=pathlib.Path("/opt/baseline-over3-20261110/ordered100-20261112");f=p/"prepared-rollout"/str(INDEX)/"prepare.json"
if f.exists():
 b=f.read_bytes();j=json.loads(b);name=str(p/"workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/row_gemv.py");changed=[n for n,h in j['sources'].items() if not pathlib.Path(n).exists() or hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()!=h];assert changed in ([name],[])
 if changed:
  assert j['sources'][name]=='618ca3568073f86f64d2873967140a6f4cf3419771c549eee50e021e4e428204' and hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()=='9b86e94c65732d1e10f4be445113418104ad42d47436272385bcf425274d9d4e'
  f.with_name('prepare.before-unrelated-sgl-source-review.json').write_bytes(b);j['sources'][name]='9b86e94c65732d1e10f4be445113418104ad42d47436272385bcf425274d9d4e';f.write_text(json.dumps(j,indent=2))
 for n,key in [('payload.pt','payload_sha256'),('case.pkl','descriptor_sha256')]:assert hashlib.sha256((f.parent/n).read_bytes()).hexdigest()==j[key]
 print(json.dumps({'ordinal':INDEX,'old_prepare_sha256':hashlib.sha256(b).hexdigest(),'new_prepare_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'changed_only_unrelated_sgl_source':changed,'payload_descriptor_sha_unchanged':True}))
else:print(json.dumps({'ordinal':INDEX,'no_previous_prepare':True}))
'''.replace('INDEX',str(i))
 r=c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True);(P/(label+'.PREPARE_SOURCE_REVIEW.json')).write_text(r.stdout)
 with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-sdk91 prepare '+label+'-prepare '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 if rc:results.append({'ordinal':i,'CPU_prepare_failed_no_device':True});continue
 while True:
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-sdk91','auto',[i],label)
  if e is not None:break
  time.sleep(30)
 if 'batch' not in e:results.append({'ordinal':i,'guard_invalid':label+'.FAILED.json'});break
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);results.append({'ordinal':i,'checkpoint':label+'.CHECKPOINT.json','result':e['batch']['results'][0]})
 (P/'MATH_TRANSFORMER_REGISTRY_PILOT_RESULTS.json').write_text(json.dumps(results,indent=2))
