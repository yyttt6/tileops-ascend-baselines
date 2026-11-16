"""Bound whole ORIGINAL GatherElements commands to exact floating3+INT32 contracts; no flags or generated JSON edits."""
from pathlib import Path
import json,subprocess,concurrent.futures,hashlib,shlex,datetime,os
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gather-v1');r=N/'source';b=r/'build-device-v1';binary=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((N/'SOURCE_MANIFEST.json').read_text())
for name,v in m['files'].items():assert sha(r/name)==v['sha256'],name
allcmds=(binary/'opc_cmd/opc_cmd.sh').read_text().splitlines();selected=[]
for cmd in allcmds:
 args=shlex.split(cmd);assert args[0]=='asc_opc';param=Path(next(x.split('=',1)[1] for x in args if x.startswith('--input_param=')));obj=json.loads(param.read_text());
 if obj['op_type']!='GatherElements':continue
 rows=obj['op_list'];assert rows
 if all(row['inputs'][0]['dtype']==row['outputs'][0]['dtype'] and row['inputs'][0]['dtype'] in ('float16','bfloat16','float32') and row['inputs'][1]['dtype']=='int32' and all(t['format']=='ND' for t in row['inputs']+row['outputs']) for row in rows):selected.append({'command':cmd,'command_sha256':hashlib.sha256(cmd.encode()).hexdigest(),'param':str(param),'param_sha256':sha(param),'op':obj['op_type']})
assert len(allcmds)>3 and len(selected)==3 and {x['op'] for x in selected}=={'GatherElements'};assert {json.loads(Path(x['param']).read_text())['op_list'][0]['inputs'][0]['dtype'] for x in selected}=={'float16','bfloat16','float32'}
params={x['param'] for x in selected};outcmds=[]
for cmd in (binary/'opc_cmd/out_cmd.sh').read_text().splitlines():
 args=shlex.split(cmd);assert args[:2]==['python3','gen_output_json.py']
 if args[2] in params:outcmds.append(cmd)
assert len(outcmds)==3;scripts=r/'scripts/kernel/binary_script';assert (scripts/'gen_output_json.py').is_file()
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_commands':len(allcmds),'selected_whole_commands':selected,'selected_original_out_cmd':outcmds,'no_flag_or_JSON_or_kernel_source_edits':True,'exact_scope':'Where5 originalstack Nx2 with native input/output FP16/BF16/FP32 andINT32 index Nx1, ND,dim1. Full originalpublicaclnnGather source selectsGatherElements on950contiguouslastaxis; no V2/otherdependency or precision substitution. Keep everywholeoriginalcommand/out_cmd for these contracts.'};(N/'GATHER_FLOAT3_I32_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n')
def execute(item):
 name=Path(item['param']).name;out=N/('compile-'+name+'.log')
 with out.open('w') as f:rc=subprocess.run(shlex.split(item['command']),cwd=binary,stdout=f,stderr=subprocess.STDOUT).returncode
 assert rc==0,(item['param'],rc);assert sha(Path(item['param']))==item['param_sha256'];return name
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for name in pool.map(execute,selected):print('ORIGINAL_GATHER_COMPILED',name,flush=True)
for cmd in outcmds:subprocess.run(shlex.split(cmd),cwd=scripts,check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for name,v in m['files'].items():assert sha(r/name)==v['sha256'],name
objects={str(p.relative_to(binary)):sha(p) for p in (binary/'ascend950').rglob('*.o')};assert len(objects)==3;proof.update(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_objects=objects,not_runtime_or_formal_acceptance=True);(N/'GATHER_FLOAT3_I32_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n');print('Gather floating3/INT32 originalcommands andoriginaldispatchmetadata complete',len(objects),flush=True)
