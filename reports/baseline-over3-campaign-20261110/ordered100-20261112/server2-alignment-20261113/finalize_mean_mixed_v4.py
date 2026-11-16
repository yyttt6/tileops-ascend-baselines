from pathlib import Path
import json,hashlib,subprocess,shlex,datetime
N=Path(__file__).resolve().parent;S=N/'source';B=S/'build';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((N/'MEAN_BF16_COMMAND_SELECTION_V2.json').read_text());r=json.loads((N/'MEAN_BF16_COMPILE_V2_RESULT.json').read_text());assert len(r['commands'])==4 and all(x['returncode']==0 for x in r['commands']);assert sha(N/'MEAN_BF16_COMMAND_SELECTION_V2.json')==r['selection_sha256']
for line in m['original_out_commands']:subprocess.run(shlex.split(line),cwd=m['cwd'],check=True)
subprocess.run(['cmake','--build',str(B),'--target','gen_bin_info_config','--parallel','1'],check=True)
objects={str(p.relative_to(B)):sha(p) for p in (B/'binary/ascend950/bin/ascend950').rglob('*.o')};assert len(objects)==8
for rel,row in json.loads((N/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(S/rel)==row['sha256'],rel
(N/'MEAN_MIXED_FINALIZE_V4.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'objects':objects,'not_coverage':True},indent=2))
