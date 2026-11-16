"""Bound whole ORIGINAL GatherElements commands to exact floating3+INT32 contracts; no flags or generated JSON edits."""
from pathlib import Path
import json,subprocess,concurrent.futures,hashlib,shlex,datetime,os
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-fp8quant-v1');r=N/'source';b=r/'build-device-v1';binary=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((N/'SOURCE_MANIFEST.json').read_text())
for name,v in m['files'].items():assert sha(r/name)==v['sha256'],name
allcmds=(binary/'opc_cmd/opc_cmd.sh').read_text().splitlines();selected=[]
for cmd in allcmds:
 args=shlex.split(cmd);assert args[0]=='asc_opc';param=Path(next(x.split('=',1)[1] for x in args if x.startswith('--input_param=')));obj=json.loads(param.read_text());
 if obj['op_type']!='DynamicBlockQuant':continue
 rows=obj['op_list'];assert rows
 if all(len(row['inputs'])==1 and len(row['outputs'])==2 and row['inputs'][0]['dtype']=='float32' and row['outputs'][0]['dtype']=='float8_e4m3fn' and row['outputs'][1]['dtype']=='float32' and all(t['format']=='ND' for t in row['inputs']+row['outputs']) for row in rows):selected.append({'command':cmd,'command_sha256':hashlib.sha256(cmd.encode()).hexdigest(),'param':str(param),'param_sha256':sha(param),'op':obj['op_type']})
assert len(allcmds)>3 and len(selected)==1 and {x['op'] for x in selected}=={'DynamicBlockQuant'};assert {json.loads(Path(x['param']).read_text())['op_list'][0]['inputs'][0]['dtype'] for x in selected}=={'float32'}
params={x['param'] for x in selected};outcmds=[]
for cmd in (binary/'opc_cmd/out_cmd.sh').read_text().splitlines():
 args=shlex.split(cmd);assert args[:2]==['python3','gen_output_json.py']
 if args[2] in params:outcmds.append(cmd)
assert len(outcmds)==1;scripts=r/'scripts/kernel/binary_script';assert (scripts/'gen_output_json.py').is_file()
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_commands':len(allcmds),'selected_whole_commands':selected,'selected_original_out_cmd':outcmds,'no_flag_or_JSON_or_kernel_source_edits':True,'exact_scope':'FP8Quant3 originalopsnn_fp8_cases FP32packed input->FLOAT8_E4M3FN codes andFP32 scales. Preservealloriginaldynamicattrs/modes/flags, fullpadding1e-4/nativeblock128or256/rint/defaultdst_type_max semantics. No kernelblock/dtype/precision/JSON edits.'};(N/'FP8QUANT_FLOAT32_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n')
def execute(item):
 name=Path(item['param']).name;out=N/('compile-'+name+'.log')
 with out.open('w') as f:rc=subprocess.run(shlex.split(item['command']),cwd=binary,stdout=f,stderr=subprocess.STDOUT).returncode
 assert rc==0,(item['param'],rc);assert sha(Path(item['param']))==item['param_sha256'];return name
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for name in pool.map(execute,selected):print('ORIGINAL_FP8QUANT_COMPILED',name,flush=True)
for cmd in outcmds:subprocess.run(shlex.split(cmd),cwd=scripts,check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for name,v in m['files'].items():assert sha(r/name)==v['sha256'],name
objects={str(p.relative_to(binary)):sha(p) for p in (binary/'ascend950').rglob('*.o')};assert len(objects)==1;proof.update(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_objects=objects,not_runtime_or_formal_acceptance=True);(N/'FP8QUANT_FLOAT32_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n');print('DynamicBlockQuant originalFP32_FP8E4M3_1 originalcommands andoriginaldispatchmetadata complete',len(objects),flush=True)
