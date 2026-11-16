"""Repair only exact uint8-wrapper runtime faults; never rerun valid or numerical failures."""
import pathlib,json,sys,fcntl,subprocess,time,datetime,hashlib
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'tla-unpack-expansion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=json.loads((P/'TASKS.json').read_text());ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());ids=[]
for k,v in ledger.items():
 r=v['result'];text=str(r)
 if r.get('key',[None])[0]=='tilelang-ascend' and r['status']!='raw_checked_pending_guard' and 'Tensor out not implemented for DT_UINT8' in text:ids.append(int(k))
ids=sorted(ids);(P/'TLA_UNPACK_EXPANSION_SCOPE.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'indices':ids,'cause':'same original uint8 arange unsupported, native comparison retained','existing_valid_skipped':True,'no_numerical_failure_retried':True},indent=2))
for i in ids:
 overlay_path=P/'RESOLVED_TLA_UNPACK.json';overlay=json.loads(overlay_path.read_text());done={e['ordinal'] for e in overlay['rows']}
 if i in done:continue
 current=json.loads((P/'ROLLOUT_LEDGER.json').read_text()).get(str(i),{}).get('result',{})
 if current.get('status')=='raw_checked_pending_guard':continue
 label='rollout-tla-unpack-candidate-'+str(i)+'-v2'
 f=P/'PINNED_DEPENDENCY_SUPERSEDED.json';j=json.loads(f.read_text());j['checkpoint_labels']=sorted(set(j['checkpoint_labels'])|{label});tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(f)
 with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare '+label+'-prepare '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError('CPU prepare rejected; inspect before next scope')
 while True:
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','tla-unpack',[i],label)
  if e is not None:break
  time.sleep(30)
 if 'batch' not in e:raise RuntimeError('Guard failure preserved, review resource before recovery')
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=e['batch']['results'][0]
 if r['status']!='raw_checked_pending_guard':
  (P/'TLA_UNPACK_EXPANSION_FAILURE.json').write_text(json.dumps({'ordinal':i,'checkpoint':label+'.CHECKPOINT.json','result':r,'no_blind_retry':True},indent=2));raise RuntimeError('Candidate failure; preserve and diagnose before expansion')
 src=P/'remote/runs'/e['tag']/'SOURCE.json';sources=json.loads(src.read_text());proof=[]
 for name in ['workspace/isolated_wrappers/tla_unpack_compat.py','tla_unpack_support.py','worker_tla_unpack.py','workspace/TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py']:
  h=hashlib.sha256((P/name).read_bytes()).hexdigest();assert sources[c.R+'/'+name]==h;proof.append({'source':name,'sha256':h,'delivery':pathlib.Path(name).name})
 overlay['rows'].append({'ordinal':i,'key':r['key'],'checkpoint':label+'.CHECKPOINT.json','result':r,'source_manifest_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_proof':proof,'guard_valid':True,'selection':'original_comparison_kernel_with_supported_unpack_wrapper','no_original_valid_timing_no_speedup_claim':True});overlay['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=overlay_path.with_suffix('.tmp');tmp.write_text(json.dumps(overlay,indent=2));tmp.replace(overlay_path)
 out=P.parent/'delivery/ordered_tla_unpack_compat/ROUTES.json';out.write_text(json.dumps({'updated_at':overlay['updated_at'],'status':'staged_not_installed','rows':overlay['rows']},indent=2));print('UNPACK_ACCEPTED',i,flush=True)
(P/'TLA_UNPACK_EXPANSION_COMPLETE.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'indices':ids,'status':'complete_scope_only','remaining_not_yet_visited_faults_not_assumed_covered':True},indent=2))
