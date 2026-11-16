"""CPU-only source packaging: unmodified original --jit host/source package."""
from pathlib import Path
import hashlib,json,shutil,subprocess,datetime
ROOT=Path('/root/dyq/workspace')
CACHE=ROOT/'.cache/all-pairs/ops_transformer/attention_build'
CACHE.mkdir(exist_ok=True)
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
source=ROOT/'ascend_baselines/ops-transformer'; target=CACHE/'source'
if not target.exists():
    shutil.copytree(source,target,ignore=lambda path,names: [n for n in names if n in ('.git','__pycache__') or (Path(path)==source and n in ('build','build_out','output'))])
(CACHE/'tmp').mkdir(exist_ok=True)
ops='flash_attn,flash_attn_metadata,flash_mla_with_kvcache,sparse_flash_attention,lightning_indexer,quant_lightning_indexer'
cmd=['bash','-c','source /root/dyq/workspace/env.sh\nexport PYTHONDONTWRITEBYTECODE=1\nexport TMPDIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/tmp\nexec bash build.sh "$@"','build','--pkg','--jit','--soc=ascend950','--ops='+ops,'--vendor_name=attentionpairs','-j8','--build-type=Release','--ccache','off']
row={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ops':ops.split(','),'command':cmd,'kernel_changes':False,'source':str(source),'copy':str(target)}
with (EVIDENCE/'attention_build.log').open('w') as f: row['returncode']=subprocess.call(cmd,cwd=target,stdout=f,stderr=subprocess.STDOUT)
row['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(EVIDENCE/'attention_build.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row),flush=True)
