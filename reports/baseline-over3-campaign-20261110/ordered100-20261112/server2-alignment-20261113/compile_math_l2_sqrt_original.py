from pathlib import Path
import json,subprocess,shlex,hashlib,concurrent.futures,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1');s=N/'source';b=N/'build-host-v2';bin=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((N/'SOURCE_MANIFEST.json').read_text())
for f,r in m['files'].items():assert sha(s/f)==r['sha256'],f
selected=[];allrows=[];wanted={'Sqrt'}
for c in (bin/'opc_cmd/opc_cmd.sh').read_text().splitlines():
 a=shlex.split(c);assert a[0]=='asc_opc';p=Path(next(v.split('=',1)[1] for v in a if v.startswith('--input_param=')));assert p.is_relative_to(N);j=json.loads(p.read_text());op=j['op_type'];types={x['dtype'] for r in j['op_list'] for x in r['inputs']+r['outputs']};r={'op':op,'dtypes':sorted(types),'command':c,'command_sha256':hashlib.sha256(c.encode()).hexdigest(),'param':str(p),'param_sha256':sha(p),'data_input_dtypes':sorted({row['inputs'][0]['dtype'] for row in j['op_list']})};allrows.append(r)
 if op=='Sqrt' and set(r['data_input_dtypes'])=={'float32'}:selected.append(r)
assert {r['op'] for r in selected}==wanted and 1<=len(selected)<=2
params={r['param'] for r in selected};outputs=[c for c in (bin/'opc_cmd/out_cmd.sh').read_text().splitlines() if shlex.split(c)[2] in params];assert len(outputs)==len(selected)
proof={'all_original_commands':allrows,'selected_whole_original_commands':selected,'original_out_commands':outputs,'source_changes':[],'params_flags_and_modes_unchanged':True,'not_runtime_acceptance':True};(N/'MATH_L2_SQRT_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n')
def run(r):
 with (N/('compile-'+r['op']+'-'+r['command_sha256'][:12]+'.log')).open('w') as out:rc=subprocess.run(shlex.split(r['command']),cwd=bin,stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,(r['op'],r['dtypes'],rc);assert sha(Path(r['param']))==r['param_sha256'];print('ORIGINAL_COMPILED',r['op'],r['dtypes'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:list(p.map(run,selected))
for c in outputs:subprocess.run(shlex.split(c),cwd=s/'scripts/kernel/binary_script',check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for f,r in m['files'].items():assert sha(s/f)==r['sha256'],f
objects={str(p.relative_to(bin)):sha(p) for p in (bin/'ascend950').rglob('*.o')};assert len(objects)==len(selected);proof.update(actual_objects=objects,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'MATH_L2_SQRT_COMPILE_SELECTION.json').write_text(json.dumps(proof,indent=2)+'\n');print('MATH_L2_SQRT_ORIGINAL_COMPILE_COMPLETE',len(objects),flush=True)
