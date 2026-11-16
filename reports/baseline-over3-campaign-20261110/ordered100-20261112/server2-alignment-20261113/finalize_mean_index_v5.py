from pathlib import Path
import json,hashlib,subprocess,shlex,datetime
N=Path(__file__).resolve().parent;S=N/'source';B=S/'build';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((N/'MEAN_BF16_COMMAND_SELECTION_V2.json').read_text());r=json.loads((N/'MEAN_BF16_COMPILE_V2_RESULT.json').read_text());assert len(r['commands'])==4 and all(x['returncode']==0 for x in r['commands']);assert sha(N/'MEAN_BF16_COMMAND_SELECTION_V2.json')==r['selection_sha256']
params={r['param'] for r in m['all_original_commands'] if r['data_input_dtypes'] in [['bfloat16'],['float32']]};assert len(params)==8
config=B/'binary/ascend950/bin/config/ascend950/reduce_mean_apt.json';backup=N/'reduce_mean_index_before_v5.json';assert not backup.exists();backup.write_bytes(config.read_bytes());config.unlink()
outs=[line for line in (B/'binary/ascend950/bin/opc_cmd/out_cmd.sh').read_text().splitlines() if shlex.split(line)[2] in params];assert len(outs)==8
for line in outs:subprocess.run(shlex.split(line),cwd=m['cwd'],check=True)
index=json.loads(config.read_text());assert len(index['binList'])==8 and {r['inputs'][0]['dtype'] for r in index['binList']}=={'float32','bfloat16'}
subprocess.run(['cmake','--build',str(B),'--target','gen_bin_info_config','--parallel','1'],check=True)
kernel_index=json.loads((B/'binary/ascend950/bin/config/ascend950/binary_info_config.json').read_text())['ReduceMean']['binaryList'];assert len(kernel_index)==8 and sum('float32' in r['binPath'] for r in kernel_index)==4
objects={str(p.relative_to(B)):sha(p) for p in (B/'binary/ascend950/bin/ascend950').rglob('*.o')};assert len(objects)==8
for rel,row in json.loads((N/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(S/rel)==row['sha256'],rel
(N/'MEAN_MIXED_FINALIZE_V5.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'objects':objects,'not_coverage':True},indent=2))
