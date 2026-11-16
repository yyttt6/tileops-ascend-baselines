"""CPU command selection or two original compiler commands per guarded invocation."""
from pathlib import Path
import json,sys,hashlib,shlex,subprocess,concurrent.futures,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-cumscan-v1');S=N/'source';B=N/'build-host-v1';bin=B/'binary/ascend950/bin';F=N/'CUMSCAN_COMPILE_SELECTION_V3.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=json.loads((N/'SOURCE_MANIFEST.json').read_text());mode=sys.argv[1]
for k,r in m['files'].items():assert sha(S/k)==r['sha256'],k
if mode=='prepare':
 assert not F.exists();allrows=[];chosen=[]
 for line in (bin/'opc_cmd/opc_cmd.sh').read_text().splitlines():
  a=shlex.split(line);assert a[0]=='asc_opc';p=Path(next(s.split('=',1)[1] for s in a if s.startswith('--input_param=')));assert p.is_relative_to(N);j=json.loads(p.read_text());dtypes=sorted({r['inputs'][0]['dtype'] for r in j['op_list']});row={'command':line,'command_sha256':hashlib.sha256(line.encode()).hexdigest(),'param':str(p),'param_sha256':sha(p),'op':j['op_type'],'data_input_dtypes':dtypes};allrows.append(row)
  if j['op_type'] in ('Cumsum','Cumprod') and len(dtypes)==1 and dtypes[0] in ['float16','bfloat16','float32']:chosen.append(row)
 assert 3<=len(chosen)<=48,(len(chosen),allrows);params={r['param'] for r in chosen};outs=[c for c in (bin/'opc_cmd/out_cmd.sh').read_text().splitlines() if shlex.split(c)[2] in params];assert len(outs)==len(chosen)
 proof={'all_original_commands':allrows,'selected_whole_original_commands':chosen,'original_out_commands':outs,'source_manifest_sha256':sha(N/'SOURCE_MANIFEST.json'),'source_changes':[],'params_flags_modes_unchanged':True,'completed_parts':[],'not_native_or_performance_acceptance':True};F.write_text(json.dumps(proof,indent=2)+'\n');print('MEAN_ORIGINAL_FP32_COMMANDS_SELECTED',len(chosen));sys.exit(0)
proof=json.loads(F.read_text());assert all(sha(r['param'])==r['param_sha256'] for r in proof['selected_whole_original_commands']);assert sha(N/'SOURCE_MANIFEST.json')==proof['source_manifest_sha256']
if mode.startswith('part'):
 part=int(mode[4:]);assert part not in proof['completed_parts'];assert proof['completed_parts']==list(range(part));rows=proof['selected_whole_original_commands'][part*2:(part+1)*2];assert 1<=len(rows)<=2
 def run(r):
  with (N/('compile-registration-v3-'+r['command_sha256'][:12]+'.log')).open('w') as out:rc=subprocess.run(shlex.split(r['command']),cwd=bin,stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,(r['param'],rc);assert sha(r['param'])==r['param_sha256'];print('ORIGINAL_COMPILED',r['op'],r['data_input_dtypes'],r['command_sha256'],flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,rows))
 proof['completed_parts'].append(part);proof['actual_objects']={str(p.relative_to(bin)):sha(p) for p in (bin/'ascend950').rglob('*.o')};assert len(proof['actual_objects'])>=min(2*(part+1),len(proof['selected_whole_original_commands']))
else:
 assert mode=='finalize' and proof['completed_parts']==list(range((len(proof['selected_whole_original_commands'])+1)//2))
 for line in proof['original_out_commands']:subprocess.run(shlex.split(line),cwd=S/'scripts/kernel/binary_script',check=True)
 subprocess.run(['cmake','--build',str(B),'--target','gen_bin_info_config','--parallel','1'],check=True);proof['actual_objects']={str(p.relative_to(bin)):sha(p) for p in (bin/'ascend950').rglob('*.o')};assert len(proof['actual_objects'])==len(proof['selected_whole_original_commands'])
 proof['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for k,r in m['files'].items():assert sha(S/k)==r['sha256'],k
F.write_text(json.dumps(proof,indent=2)+'\n');print('MEAN_ORIGINAL_STAGE_COMPLETE',mode,flush=True)
