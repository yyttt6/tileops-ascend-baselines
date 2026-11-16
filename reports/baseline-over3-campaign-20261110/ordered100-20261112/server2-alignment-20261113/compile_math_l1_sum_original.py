from pathlib import Path
import json,hashlib,shlex,subprocess,datetime,concurrent.futures
n=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-countnz-v1');b=n/'build-host-v1';s=n/'source';d=b/'binary/ascend950/bin';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((n/'COUNTNZ_COMPILE_SELECTION.json').read_text());selected=[];libs={str(p):sha(p) for p in b.glob('*.so')}
for f,r in json.loads((n/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(s/f)==r['sha256']
for f,h in old['actual_objects'].items():assert sha(d/f)==h
for c in (d/'opc_cmd/opc_cmd.sh').read_text().splitlines():
 a=shlex.split(c);p=Path(next(v.split('=',1)[1] for v in a if v.startswith('--input_param=')));j=json.loads(p.read_text());t=sorted({r['inputs'][0]['dtype'] for r in j['op_list']})
 if j['op_type']=='ReduceSum' and set(t)<={'float16','bfloat16'}:selected.append({'op':'ReduceSum','data_input_dtypes':t,'param':str(p),'param_sha256':sha(p),'command':c,'command_sha256':hashlib.sha256(c.encode()).hexdigest()})
assert len(selected)==4 and {x for r in selected for x in r['data_input_dtypes']}=={'float16','bfloat16'}
def run(r):
 with (n/('mathl1-'+r['command_sha256'][:12]+'.log')).open('w') as out:rc=subprocess.run(shlex.split(r['command']),cwd=d,stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,(r['data_input_dtypes'],rc);assert sha(Path(r['param']))==r['param_sha256'];print('ORIGINAL_L1_SUM_COMPILED',r['data_input_dtypes'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,selected))
params={r['param'] for r in selected};outs=[c for c in (d/'opc_cmd/out_cmd.sh').read_text().splitlines() if shlex.split(c)[2] in params];assert len(outs)==4
for c in outs:subprocess.run(shlex.split(c),cwd=s/'scripts/kernel/binary_script',check=True)
subprocess.run(['cmake','--build',str(b),'--target','gen_bin_info_config','--parallel','1'],check=True)
for f,h in old['actual_objects'].items():assert sha(d/f)==h
for f,h in libs.items():assert sha(Path(f))==h
for f,r in json.loads((n/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(s/f)==r['sha256']
objects={str(p.relative_to(d)):sha(p) for p in (d/'ascend950').rglob('*.o')};assert len(objects)==15;v={'new_whole_original_commands':selected,'selected_whole_original_commands':old['selected_whole_original_commands']+selected,'preserved_old11_objects':old['actual_objects'],'preserved_host_API_proto_libraries':libs,'actual_objects':objects,'source_changes':[],'original_flags_params_modes':True,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};(n/'MATH_L1_SUM_COMPILE_SELECTION.json').write_text(json.dumps(v,indent=2)+'\n');print('ORIGINAL_L1_SUM4_COMPLETE_PRESERVED_OLD11_HOSTLIBS',flush=True)
