"""CPU-only bind exact private JIT source, ELF metadata, guard PID and actual launches."""
from pathlib import Path
import json,hashlib,datetime,re,subprocess,shlex,tarfile
A=Path(__file__).resolve().parent;N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-selection-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for num,host in [(1,'openlibing-DevEnv-661495'),(2,'openlibing-DevEnv-055409')]:
 state=json.loads((A/f'SERVER{num}_TRANSFORMER_SELECTION_PROBE_V1.json').read_text());assert state.get('exit_code')==0,state;tag=state['tag'];g=json.loads((A/(tag+'-guard.json')).read_text());assert g['phase']=='finished' and g['performance_valid'] and g['workload_exit_code']==0;pid=g['child']['pid'];log=A/(tag+'.log');lines=log.read_text(errors='replace').splitlines();compiles=[]
 for line in lines:
  if 'LogTaskDetails' not in line or 'opType[MoeGatingTopKSoftmaxV2]' not in line or 'result[success]' not in line:continue
  kernel=re.search(r'kernel\[([^]]+)\]',line)[1];file=re.search(r'File\[([^]]+)\]',line)[1];partial=line.split('json file path[',1)[1];parent=str(Path(partial).parent);launch=[l for l in lines if f'RUNTIME({pid},' in l and 'StreamLaunchKernelV1:' in l and 'kernel_name='+kernel+'__kernel0,' in l];assert len(launch)==1,(kernel,len(launch));compiles.append({'kernel':kernel,'source':file,'cache_parent_from_truncated_log':parent,'full_cache_basename_reconstructed_from_fully_logged_kernel':True,'compile_log':line,'actual_launch_log':launch[0]})
 assert len(compiles)==2;code=r'''from pathlib import Path
import json,sys,hashlib,shutil,tarfile,datetime
arg=json.loads(sys.stdin.buffer.read());n=Path(arg['N']);tag=arg['tag'];m=json.loads((n/'TRANSFORMER_SELECTION_PACKAGE.json').read_text());v=Path(m['vendor']);root=n/'qualification-artifacts'/tag;root.mkdir(parents=True,exist_ok=False);files={};proof=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for item in arg['compiles']:
 p=Path(item['source']);rel=str(p.relative_to(v));assert sha(p)==m['files'][rel]['sha256'];k=item['kernel'];parent=Path(item['cache_parent_from_truncated_log']);meta=parent/(k+'.json');o=parent/(k+'.o');data=json.loads(meta.read_text());assert data['sha256']==sha(o),(str(o),data.get('sha256'));copies=[]
 for p in parent.glob(k+'*'):
  if p.is_file():
   dst=root/p.name;shutil.copy2(p,dst);copies.append({'name':p.name,'sha256':sha(dst),'bytes':dst.stat().st_size});files[p.name]=sha(dst)
 proof.append({'kernel':k,'private_source':item['source'],'private_entry_source_sha256':m['files'][rel]['sha256'],'device_ELF_sha256':sha(o),'metadata':data,'cache_files':copies})
result=n/(tag+'.json');data=json.loads(result.read_text());assert data['status']=='full_original_V2_graph_API_fixtures_passed_pending_private_JIT_attribution' and len(data['checks'])==2
archive=n/(tag+'.jit.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in root.iterdir():t.add(p,arcname=p.name,recursive=False)
r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':arg['host'],'guard_owner':arg['owner'],'device_node':arg['device_node'],'private_original_JIT_sources_ELF_and_actual_launches_verified':True,'full3output_fixtures':data['checks'],'probe_result_sha256':sha(result),'manifest_sha256':sha(n/'TRANSFORMER_SELECTION_PACKAGE.json'),'source_pin':m['source_pin'],'proof':proof,'jit_archive':str(archive),'jit_archive_sha256':sha(archive),'files':files,'source_or_kernel_or_compilerflags_edits':False};(n/'PRIVATE_SELECTION_QUALIFICATION.json').write_text(json.dumps(r,indent=2)+chr(10));print(json.dumps(r))'''
 arg={'N':N,'tag':tag,'host':host,'owner':g['owner'],'device_node':g['device_node'],'compiles':compiles};r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',host,'python3 -I -S -c '+shlex.quote(code)],input=json.dumps(arg).encode(),capture_output=True,timeout=45);assert r.returncode==0,r.stderr.decode();proof=json.loads(r.stdout);result_name=tag+'.json'
 if not (A/result_name).exists():
  data=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes',host,'cat '+shlex.quote(N+'/'+result_name)],capture_output=True,check=True).stdout;(A/result_name).write_bytes(data)
 assert sha(A/result_name)==proof['probe_result_sha256'];proof.update(guard_receipt=tag+'-guard.json',guard_receipt_sha256=sha(A/(tag+'-guard.json')),log_sha256=sha(log),compile_launch_evidence=compiles)
 target=A/(tag+'.jit.tar.gz');r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes',host,'cat '+shlex.quote(proof['jit_archive'])],capture_output=True,check=True);target.write_bytes(r.stdout);assert sha(target)==proof['jit_archive_sha256'];p=A/f'PRIVATE_TRANSFORMER_SELECTION_QUALIFICATION_HOST{num}.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'host':num,'status':'private_JITsource_ELF_actuallaunch_full3outputs_guard_verified','kernels':len(proof['proof']),'archive':target.name}))
