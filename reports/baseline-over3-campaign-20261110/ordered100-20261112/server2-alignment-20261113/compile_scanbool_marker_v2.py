from pathlib import Path
import json,hashlib,subprocess,shlex,datetime
n=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1');b=n/'build-host-v1';s=n/'source';d=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((n/'SCANBOOL_BOOL4_COMPILE_SELECTION.json').read_text());selected=[]
for f,r in json.loads((n/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(s/f)==r['sha256']
for f,h in old['actual_objects'].items():assert sha(d/f)==h
for c in (d/'opc_cmd/opc_cmd.sh').read_text().splitlines():
 a=shlex.split(c);p=Path(next(x.split('=',1)[1] for x in a if x.startswith('--input_param=')));j=json.loads(p.read_text());t={r['inputs'][0]['dtype'] for r in j['op_list']}
 if j['op_type']=='NotEqual' and t=={'int32'}:selected.append({'op':'NotEqual','dtypes':sorted(t),'param':str(p),'param_sha256':sha(p),'command':c,'command_sha256':hashlib.sha256(c.encode()).hexdigest()})
assert len(selected)==1
for r in selected:
 with (n/'compile-NotEqual-INT32-marker-v2.log').open('w') as out:rc=subprocess.run(shlex.split(r['command']),cwd=d,stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0;assert sha(Path(r['param']))==r['param_sha256']
params={r['param'] for r in selected};outputs=[c for c in (d/'opc_cmd/out_cmd.sh').read_text().splitlines() if shlex.split(c)[2] in params];assert len(outputs)==1
for c in outputs:subprocess.run(shlex.split(c),cwd=s/'scripts/kernel/binary_script',check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for f,h in old['actual_objects'].items():assert sha(d/f)==h
objects={str(p.relative_to(d)):sha(p) for p in (d/'ascend950').rglob('*.o')};assert len(objects)==3
v={'selected_whole_original_commands':old['selected_whole_original_commands']+selected,'new_INT32_command':selected,'preserved_old_objects':old['actual_objects'],'actual_objects':objects,'source_changes':[],'all_flags_params_modes_original':True,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};(n/'SCANBOOL_MARKER_V2_COMPILE_SELECTION.json').write_text(json.dumps(v,indent=2)+'\n');print('ORIGINAL_INT32_MARKER_COMPILED_PRESERVED_BOOL2',flush=True)
