"""Bounded CPU handoff; no device use outside each canonical child guard."""
from pathlib import Path
import json,os,time,sys,subprocess,datetime,hashlib,fcntl
P=Path(__file__).resolve().parent;now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
l=(P/'clamp-small-handoff.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'waiting_representative_and_CPU_inputs','counts_as_primary_dispatch':False}
def put():
 s['checked_at']=now();p=P/'CLAMP_SMALL_FORMAL_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
try:
 deadline=time.monotonic()+1200;put()
 while True:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
  e=json.loads((P/'OPS_NN_CLAMP_FOREACH_EXPANSION.json').read_text());assert e['status']!='stopped_requires_diagnosis','representative fault retained; no retry'
  cpu=json.loads((P/'OPS_NN_CLAMP_FOREACH_INPUTS.json').read_text());assert cpu['status']!='CPU_source_input_fault_requires_diagnosis','CPU source/input fault'
  cur=json.loads((P/'CURRENT_PROCESS.json').read_text())
  if e['status']=='representative1_fullchecker15raw_guard_valid_expand_only_after_review' and cpu.get('verified_inputs')==5 and cur['status']=='group_finished_next_dispatch_required':break
  assert time.monotonic()<deadline,'bounded waitexpired';put();time.sleep(10)
 p=P/'OPS_NN_CLAMP_FOREACH_CURRENT.json';r=json.loads(p.read_text());assert r['accepted_original_APIs']==1 and r['rows'][0]['ordinal']==435 and r['rows'][0]['guard_checker_15raw_valid']
 text=(P/'queue_ops_nn_clamp_foreach.py').read_text().replace("'scope':[435]","'scope':[431,432,435,436,439,440]").replace("'accepted_original_APIs':0","'accepted_original_APIs':1").replace('for ids in [[435]]:','for ids in [[431,432],[436],[439,440]]:').replace('representative1_fullchecker15raw_guard_valid_expand_only_after_review','all6_small_exact_original_APIs_fullchecker15raw_guard_valid')
 text=text.replace(' assert not stop();q='," prior_current=json.loads((P/'OPS_NN_CLAMP_FOREACH_CURRENT.json').read_text());assert prior_current['accepted_original_APIs']==1 and prior_current['rows'][0]['ordinal']==435\n assert not stop();q=")
 out=P/'queue_ops_nn_clamp_foreach_small_expand.py';assert not out.exists();compile(text,str(out),'exec');out.write_text(text)
 (P/'CLAMP_FOREACH_SMALL_EXPANSION_REVIEW.json').write_text(json.dumps({'checked_at':now(),'representative_summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'valid_skip':[435],'new_scope':[431,432,436,439,440],'CPU_inputs_source_verified':5,'original_dtype_timed_AST_nativeAPI_and_checker_unchanged':True,'other256M4_unattempted_not_rejected_by_prediction':True,'controller_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+'\n')
 s.update(status='running_formal_child',child=out.name);put()
 with (P/'ops-nn-clamp-foreach-small-expand.log').open('a') as f:subprocess.run([sys.executable,'-u',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.update(status='small6_qualified_next_dispatch_required')
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:put()
