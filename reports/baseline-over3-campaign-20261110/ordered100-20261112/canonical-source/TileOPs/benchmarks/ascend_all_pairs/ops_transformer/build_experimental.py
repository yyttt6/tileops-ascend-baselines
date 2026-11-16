"""CPU-only deployment of unchanged original experimental Mamba frontends."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
R=Path('/root/dyq/workspace');E=R/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer';C=R/'.cache/all-pairs/ops_transformer'
name=sys.argv[1]
assert name in ('chunk_scan','chunk_cumsum')
base=json.loads((E/'state_passing_build.json').read_text());source=R/f'ascend_baselines/ops-transformer/experimental/mamba/mamba2_{name}/torch_interface.cpp';out=C/(name+'_build');out.mkdir(exist_ok=True)
oldsrc=base['source'];oldobj=str(C/'state_passing_build/original.o');oldbin=str(C/'state_passing_build/liboriginal_state_passing.so')
newobj=str(out/'original.o');newbin=str(out/f'liboriginal_{name}.so')
replace={oldsrc:str(source),oldobj:newobj,oldbin:newbin}
compile=[replace.get(a,a) for a in base['compile']];link=[replace.get(a,a) for a in base['link']]
record={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compile':compile,'link':link,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_source_modified':False,'cpu_only':True}
with (E/(name+'_build.log')).open('w') as f:
 record['compile_returncode']=subprocess.run(compile,stdout=f,stderr=subprocess.STDOUT,cwd=out).returncode
 if record['compile_returncode']==0:record['link_returncode']=subprocess.run(link,stdout=f,stderr=subprocess.STDOUT,cwd=out).returncode
if record.get('link_returncode')==0:record['binary_sha256']=hashlib.sha256(Path(newbin).read_bytes()).hexdigest()
record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(E/(name+'_build.json')).write_text(json.dumps(record,indent=2)+'\n');print(record.get('compile_returncode'),record.get('link_returncode'))
