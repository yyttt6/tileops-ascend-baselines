from pathlib import Path
import json,subprocess,concurrent.futures,hashlib,shlex,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-realdiv-v1');s=N/'source';b=N/'build-host-v1';binary=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((N/'SOURCE_MANIFEST.json').read_text())
for name,v in m['files'].items():assert sha(s/name)==v['sha256'],name
commands=(binary/'opc_cmd/opc_cmd.sh').read_text().splitlines();selected=[]
for cmd in commands:
 args=shlex.split(cmd);assert args[0]=='asc_opc';param=Path(next(v.split('=',1)[1] for v in args if v.startswith('--input_param=')));obj=json.loads(param.read_text());rows=obj['op_list']
 if obj['op_type']=='RealDiv' and all(len(r['inputs'])==2 and len(r['outputs'])==1 and len({x['dtype'] for x in r['inputs']+r['outputs']})==1 and r['inputs'][0]['dtype'] in {'float32'} and all(x['format']=='ND' for x in r['inputs']+r['outputs']) for r in rows):selected.append({'command':cmd,'sha256':hashlib.sha256(cmd.encode()).hexdigest(),'param':str(param),'param_sha256':sha(param),'dtype':rows[0]['inputs'][0]['dtype']})
assert len(commands)>=2 and len(selected)==2 and {x['dtype'] for x in selected}=={'float32'}
params={x['param'] for x in selected};outcmds=[c for c in (binary/'opc_cmd/out_cmd.sh').read_text().splitlines() if shlex.split(c)[2] in params];assert len(outcmds)==2
proof={'original_commands_count':len(commands),'selected_whole_original_commands':selected,'original_out_cmd':outcmds,'no_kernel_flags_param_source_edits':True,'no_runtime_or_performance_acceptance':True};(N/'REALDIVFLOAT32_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n')
def run(x):
 with (N/('compile-'+x['dtype']+'-'+x['sha256'][:12]+'.log')).open('w') as out:r=subprocess.run(shlex.split(x['command']),cwd=binary,stdout=out,stderr=subprocess.STDOUT)
 assert r.returncode==0,(x['dtype'],r.returncode);assert sha(Path(x['param']))==x['param_sha256'];print('ORIGINAL_REALDIV_COMPILED',x['dtype'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,selected))
for cmd in outcmds:subprocess.run(shlex.split(cmd),cwd=s/'scripts/kernel/binary_script',check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for name,v in m['files'].items():assert sha(s/name)==v['sha256'],name
objects={str(p.relative_to(binary)):sha(p) for p in (binary/'ascend950').rglob('*.o')};assert len(objects)==2;proof.update(actual_objects=objects,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'REALDIVFLOAT32_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n');print('ORIGINAL_REALDIVFLOAT2_COMPLETE',flush=True)
