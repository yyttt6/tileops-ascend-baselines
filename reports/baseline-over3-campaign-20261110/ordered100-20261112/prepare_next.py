"""CPU-only preparation, grouped once per operator, foreground child processes."""
import pathlib,json,subprocess,itertools,time
P=pathlib.Path(__file__).resolve().parent;R='/opt/baseline-over3-20261110/ordered100-20261112';H='openlibing-DevEnv-661495';rows=json.loads((P/'TASKS.json').read_text())
for op,items in itertools.groupby([r for r in rows if r['library']=='TileKernels' and r['ordinal']>=10],key=lambda r:r['op']):
 ids=[r['ordinal'] for r in items]
 for start in range(0,len(ids),8):
  chunk=ids[start:start+8];tag='prep-'+str(chunk[0]);t=time.perf_counter()
  with (P/(tag+'.log')).open('w') as f:rc=subprocess.run(['ssh',H,'bash '+R+'/launch.sh template prepare '+tag+' '+' '.join(map(str,chunk))],stdout=f,stderr=subprocess.STDOUT).returncode
  with (P/'CPU_PREP_TIMING.jsonl').open('a') as f:f.write(json.dumps({'op':op,'ids':chunk,'exit_code':rc,'seconds':time.perf_counter()-t})+'\n')
  print('CPU',chunk,'exit',rc,round(time.perf_counter()-t,2),flush=True)
  if rc:break
