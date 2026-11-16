from pathlib import Path
import json,hashlib,subprocess,shlex,concurrent.futures,datetime,os
N=Path(__file__).resolve().parent
p=N/'MEAN_BF16_COMMAND_SELECTION_V2.json';m=json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert m['original_params_flags_modes_unchanged'] and len(m['selected_original_BF16_commands'])==4
source=json.loads((N/'SOURCE_MANIFEST.json').read_text())
for rel,r in source['files'].items():assert sha(N/'source'/rel)==r['sha256'],rel
assert sha(N/'SOURCE_MANIFEST.json')==m['source_manifest_sha256']
for r in m['selected_original_BF16_commands']:
 assert sha(r['param'])==r['param_sha256']
 assert hashlib.sha256(r['command'].encode()).hexdigest()==r['command_sha256']
def run(r):
 log=N/('bf16-selected-v2-'+Path(r['param']).name+'.log')
 assert not log.exists(),str(log)
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with log.open('w') as f:rc=subprocess.run(shlex.split(r['command']),cwd=m['cwd'],stdout=f,stderr=subprocess.STDOUT).returncode
 row={'command_sha256':r['command_sha256'],'param_sha256':r['param_sha256'],'started_at':started,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':rc,'log_sha256':sha(log)}
 (N/(log.name+'.receipt.json')).write_text(json.dumps(row,indent=2))
 assert rc==0,row
 return row
rows=[]
# Each wave waits for both same-process-group foreground commands; no background escape.
for i in (0,2):
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows.extend(pool.map(run,m['selected_original_BF16_commands'][i:i+2]))
(N/'MEAN_BF16_COMPILE_V2_RESULT.json').write_text(json.dumps({'commands':rows,'source_manifest_sha256':m['source_manifest_sha256'],'selection_sha256':sha(p),'not_coverage':True},indent=2))
print('Four exact original BF16 commands finished; packaging and qualification still required',flush=True)
