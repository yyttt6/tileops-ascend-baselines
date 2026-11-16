"""Bound complete ORIGINAL command entries to reviewed FP32 native10 scope; do not modify flags or generated JSON."""
from pathlib import Path
import json,subprocess,concurrent.futures,hashlib,shlex,datetime,os
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1');old=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1');r=old/'source';b=r/'build-device-v1';binary=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((old/'SOURCE_MANIFEST.json').read_text())
for name,v in m['files'].items():assert sha(r/name)==v['sha256'],name
allcmds=(binary/'opc_cmd/opc_cmd.sh').read_text().splitlines();selected=[]
for cmd in allcmds:
 args=shlex.split(cmd);assert args[0]=='asc_opc';param=Path(next(x.split('=',1)[1] for x in args if x.startswith('--input_param=')));obj=json.loads(param.read_text());assert obj['op_type'] in ('MatMulV3','BatchMatMulV3','FusedMatMul')
 rows=obj['op_list'];assert rows
 if all(row['inputs'][0]['dtype']==row['inputs'][1]['dtype']==row['outputs'][0]['dtype'] and row['outputs'][0]['dtype'] in ('float16','bfloat16') for row in rows):selected.append({'command':cmd,'command_sha256':hashlib.sha256(cmd.encode()).hexdigest(),'param':str(param),'param_sha256':sha(param),'op':obj['op_type']})
assert len(allcmds)==42 and len(selected)==18 and {x['op'] for x in selected}=={'MatMulV3','BatchMatMulV3','FusedMatMul'}
params={x['param'] for x in selected};outcmds=[]
for cmd in (binary/'opc_cmd/out_cmd.sh').read_text().splitlines():
 args=shlex.split(cmd);assert args[:2]==['python3','gen_output_json.py']
 if args[2] in params:outcmds.append(cmd)
assert len(outcmds)==18;scripts=r/'scripts/kernel/binary_script';assert (scripts/'gen_output_json.py').is_file()
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_commands':42,'selected_whole_commands':selected,'selected_original_out_cmd':outcmds,'no_flag_or_JSON_or_kernel_source_edits':True,'native_original_dtype_scope':'Only ordinals490/494/496/497; fulloriginal1024x256x64pack, nativeF16/BF16output; no FP32widening, no sameconditionreferencefailure retry; all18 completeoriginalMatMulV3/BatchMatMulV3/FusedMatMul F16/BF16commands andmodes retained.'};(N/'PAD4_COMPILE_SELECTION_H1.json').write_text(json.dumps(proof,indent=2)+'\n')
def execute(item):
 name=Path(item['param']).name;out=N/('compile-'+name+'.log')
 with out.open('w') as f:rc=subprocess.run(shlex.split(item['command']),cwd=binary,stdout=f,stderr=subprocess.STDOUT).returncode
 assert rc==0,(item['param'],rc);assert sha(Path(item['param']))==item['param_sha256'];return name
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for name in pool.map(execute,selected):print('ORIGINAL_F16_BF16_COMPILED',name,flush=True)
for cmd in outcmds:subprocess.run(shlex.split(cmd),cwd=scripts,check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for name,v in m['files'].items():assert sha(r/name)==v['sha256'],name
objects={str(p.relative_to(binary)):sha(p) for p in (binary/'ascend950').rglob('*.o')};assert len(objects)>=4;proof.update(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_objects=objects,not_runtime_or_formal_acceptance=True);(N/'PAD4_COMPILE_SELECTION_H1.json').write_text(json.dumps(proof,indent=2)+'\n');print('F16BF16 originalcommands andoriginaldispatchmetadata complete',len(objects),flush=True)
