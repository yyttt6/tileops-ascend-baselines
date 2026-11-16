"""CPU-only bounded handoff; each device batch remains under its original guard."""
from pathlib import Path
import json,time,subprocess,sys,datetime,hashlib,os,fcntl
P=Path(__file__).resolve().parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
lock=(P/'dequant-block-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'waiting_channel7_terminal','checked_at':now(),'counts_as_primary_dispatch':False,'no_parallel_device_work':True}
def put():
 s['checked_at']=now();p=P/'DEQUANT_BLOCK_FORMAL_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
def stop():assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
def run(entry,log):
 stop();s.update(status='running_child',child_entry=entry);put()
 with (P/log).open('a') as f:subprocess.run([sys.executable,'-u',str(P/entry)],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 put();deadline=time.monotonic()+1200
 while True:
  stop();e=json.loads((P/'OPS_NN_DEQUANT_CHANNEL_FOREACH_EXPANSION.json').read_text());assert e['status']!='stopped_requires_diagnosis','predecessor fault; no blind continuation'
  if e['status']=='all7_exact_original_APIs_fullchecker15raw_guard_valid':
   r=json.loads((P/'OPS_NN_DEQUANT_CHANNEL_FOREACH_CURRENT.json').read_text());assert r['accepted_original_APIs']==7 and all(x['guard_checker_15raw_valid'] for x in r['rows'])
   cur=json.loads((P/'CURRENT_PROCESS.json').read_text())
   if cur['status']=='group_finished_next_dispatch_required':break
  assert time.monotonic()<deadline,'bounded predecessor wait expired';put();time.sleep(10)
 s['predecessor_summary_sha256']=hashlib.sha256((P/'OPS_NN_DEQUANT_CHANNEL_FOREACH_CURRENT.json').read_bytes()).hexdigest();put()
 run('queue_native_dequant_block_foreach_inputs.py','native-dequant-block-foreach-inputs.log')
 run('queue_ops_nn_dequant_block_foreach.py','ops-nn-dequant-block-foreach-queued.log')
 p=P/'OPS_NN_DEQUANT_BLOCK_FOREACH_CURRENT.json';r=json.loads(p.read_text());assert r['accepted_original_APIs']==1 and r['rows'][0]['ordinal']==503 and r['rows'][0]['guard_checker_15raw_valid']
 text=(P/'queue_ops_nn_dequant_block_foreach.py').read_text().replace("'scope':[503]","'scope':[499,500,501,502,503,504]").replace("'accepted_original_APIs':0","'accepted_original_APIs':1").replace('for ids in [[503]]:','for ids in [[501,504],[502],[500],[499]]:').replace('representative1_fullchecker15raw_guard_valid_expand_only_after_review','all6_exact_original_APIs_fullchecker15raw_guard_valid')
 text=text.replace(' assert not stop();q='," prior_current=json.loads((P/'OPS_NN_DEQUANT_BLOCK_FOREACH_CURRENT.json').read_text());assert prior_current['accepted_original_APIs']==1 and prior_current['rows'][0]['ordinal']==503\n assert not stop();q=")
 out=P/'queue_ops_nn_dequant_block_foreach_expand.py';assert not out.exists();compile(text,str(out),'exec');out.write_text(text)
 (P/'DEQUANT_BLOCK_FOREACH_EXPANSION_REVIEW.json').write_text(json.dumps({'checked_at':now(),'representative_summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'valid_skip':[503],'new_scope':[501,504,502,500,499],'original_timed_AST_native_reference_checker_unchanged':True,'controller_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+'\n')
 run(out.name,'ops-nn-dequant-block-foreach-expand.log');r=json.loads(p.read_text());assert r['accepted_original_APIs']==6;s.update(status='block6_all_qualified_next_dispatch_required',accepted=6)
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:put()
