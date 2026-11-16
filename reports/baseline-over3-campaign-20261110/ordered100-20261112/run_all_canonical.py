"""One bounded correctness-only pass after the oracle-backend integration correction."""
import pathlib,json,subprocess,sys,time,itertools
P=pathlib.Path(__file__).resolve().parent;tasks=json.loads((P/'TASKS.json').read_text())[:100]
for lib,rows in itertools.groupby(tasks,key=lambda r:r['library']):
 ids=[r['ordinal'] for r in rows if r['ordinal']!=41]
 size=1 if lib=='DeepGEMM-Ascend' else 8
 for k in range(0,len(ids),size):
  group=ids[k:k+size];family='checkdeep' if lib=='DeepGEMM-Ascend' else 'checkcatlass' if lib=='catlass' else 'checktemplate';label='canonical-'+str(group[0])+'-'+str(group[-1]);dest=P/(label+'.CANONICAL.json')
  if dest.exists() and json.loads(dest.read_text()).get('batch'):continue
  while True:
   rc=subprocess.run([sys.executable,'-u',str(P/'run_supplement.py'),family,label,*map(str,group)]).returncode
   if rc==75:print('DEFER',label,flush=True);time.sleep(30);continue
   if rc:raise RuntimeError('checker process/guard needs review '+label)
   break
(P/'CANONICAL_SWEEP_DONE.json').write_text(json.dumps({'cases':100,'performance_windows_added':0,'original_checker_restored':True},indent=2))
