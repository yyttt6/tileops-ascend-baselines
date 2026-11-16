"""Resume <=2 exact compiler commands per guard; immutable completed bytes and original flags."""
from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,shlex,sys,datetime,os,fcntl
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1');old=N.parent/'ops-nn-unary-gemm10-v1';source=old/'source';b=source/'build-device-v1/binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();file=N/'PAD4_COMPILE_RESUME_LEDGER.json';lock=N/'compiler-resume.lock';g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['child']['pid']==os.getpgrp() and g['owner'].startswith('root/baseline-native-build/')
with lock.open('a') as l:
 fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=json.loads(file.read_text());plan=json.loads((N/'PAD4_COMPILE_SELECTION_H1.json').read_text());assert sha(N/'PAD4_COMPILE_SELECTION_H1.json')==ledger['original_selection_sha256'];sm=json.loads((old/'SOURCE_MANIFEST.json').read_text());assert sha(old/'SOURCE_MANIFEST.json')==ledger['source_manifest_sha256'];assert all(sha(source/k)==v['sha256'] for k,v in sm['files'].items())
 def save():
  ledger['checked_at']=now();t=file.with_suffix('.next.tmp');t.write_text(json.dumps(ledger,indent=2)+'\n');t.replace(file)
 def verify_done():
  for h,v in ledger['compiled'].items():
   assert h==v['command']['command_sha256'] and sha(v['command']['param'])==v['command']['param_sha256']
   for p,r in v['artifacts'].items():assert sha(p)==r['sha256'],p
 verify_done()
 if sys.argv[1]=='finalize':
  assert len(ledger['compiled'])==len(plan['selected_whole_commands'])==18
  scripts=source/'scripts/kernel/binary_script'
  for cmd in plan['selected_original_out_cmd']:subprocess.run(shlex.split(cmd),cwd=scripts,check=True)
  subprocess.run(['cmake','--build',str(source/'build-device-v1'),'--target','gen_bin_info_config','--parallel','1'],check=True)
  verify_done();assert all(sha(source/k)==v['sha256'] for k,v in sm['files'].items());objects={str(p.relative_to(b)):sha(p) for p in (b/'ascend950').rglob('*.o')};assert len(objects)==18
  plan.update(completed_at=now(),actual_objects=objects,not_runtime_or_formal_acceptance=True,original_selection_sha256=ledger['original_selection_sha256'],resume_ledger_sha256=sha(file),no_completed_command_rerun=True);(N/'PAD4_COMPILE_COMPLETED_H1.json').write_text(json.dumps(plan,indent=2)+'\n');print('FINALIZED18_ORIGINAL_COMMANDS',flush=True)
 else:
  ids=list(map(int,sys.argv[1:]));assert 0<len(ids)<=2
  selected=[plan['selected_whole_commands'][i] for i in ids];assert all(v['command_sha256'] not in ledger['compiled'] for v in selected),'Never rerun completed command'
  def run(row):
   assert hashlib.sha256(row['command'].encode()).hexdigest()==row['command_sha256'] and sha(row['param'])==row['param_sha256'];log=N/('resume-'+Path(row['param']).name+'.log')
   assert not log.exists(),'A prior samevariant resume attempt needs explicit diagnosis'
   with log.open('w') as out:r=subprocess.run(shlex.split(row['command']),cwd=b,stdout=out,stderr=subprocess.STDOUT)
   assert r.returncode==0,(row['param'],r.returncode)
   data=json.loads(Path(row['param']).read_text());files={}
   for op in data['op_list']:
    found=list((b/'ascend950').rglob(op['bin_filename']+'.o'));assert len(found)==1 and found[0].read_bytes()[:4]==bytes([127,69,76,70])
    for p in [found[0],found[0].with_suffix('.json')]:assert p.is_file();files[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
   return row['command_sha256'],{'command':row,'artifacts':files,'phase':'individual_compiler_exit0_guard_pending','completed_at':now(),'compile_log_sha256':sha(log),'guard_receipt_owner':g['owner'],'not_runtime_or_performance_acceptance':True}
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(run,row) for row in selected]
   for future in concurrent.futures.as_completed(futures):h,result=future.result();ledger['compiled'][h]=result;save();print('COMPILER_COMMAND_CHECKPOINT',Path(result['command']['param']).name,len(ledger['compiled']),flush=True)
  assert all(sha(source/k)==v['sha256'] for k,v in sm['files'].items());verify_done()
