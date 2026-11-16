"""Offline functional/identity audit plus dualarchive SHA, never device work."""
from pathlib import Path
import json,re,hashlib,subprocess,shlex,tarfile,datetime
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-adaln3-v1';H='openlibing-DevEnv-661495';s=json.loads((A/'SERVER1_MATH_ADALN3_QUALIFICATION.json').read_text());tag=s['tag'];g=json.loads((A/(tag+'-guard.json')).read_text());r=json.loads((A/(tag+'.json')).read_text());cpu=json.loads((P/'OPS_MATH_ADALN3_CPU_PROOF.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def remote(cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',H,cmd],check=True,**kw)
assert g['phase']=='finished' and g['performance_valid'] is True and g['workload_exit_code']==0
assert r['full_output_and_inputs_verified'] and r['original_1GiB_flush_verified'] and r['component_source_sha256_unchanged'] and len(r['checks'])==4 and len(cpu['checks'])==12 and cpu['no_NPU_imported']
for name,h in cpu['sources'].items():assert sha(P/name)==h
log=(A/(tag+'.log')).read_text();counts={}
for line in log.splitlines():
 if 'StreamLaunchKernelV2:' in line:
  m=re.search(r'kernel_name=([^,]+)',line)
  if m and (m[1].startswith(('Mul_','Add_')) or 'layer_norm_fp32' in m[1]):counts[m[1]]=counts.get(m[1],0)+1
assert sum(v for k,v in counts.items() if 'layer_norm_fp32' in k)==4
code='import json,pathlib;files='+repr(list(r['component_sources_sha256']))+';print(json.dumps({f:str(pathlib.Path(f).resolve()) for f in files}))';canonical=json.loads(remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True).stdout)
objects={}
for prefix,expected in [('Mul_','/ops-math-dequant-v1/'),('Add_','/ops-math-add-v1/complete-private-device-v2/')]:
 names=[k for k,v in counts.items() if k.startswith(prefix) and v==4];assert len(names)==1
 basename=names[0].rsplit('_',1)[0]+'.o';matches={k:v for k,v in r['component_sources_sha256'].items() if Path(k).name==basename and expected in canonical[k]};assert len(matches)==1 and all(canonical[k] in log for k in matches);objects.update(matches)
maps=(A/(tag+'.json.maps.txt')).read_text();assert '/ops-math-layernorm-original-v1/libops_math_layernorm.so' in maps and '/ops-math-dequant-v1/' in maps and '/ops-math-add-v1/complete-private-device-v2/' in maps
code='import json,pathlib,hashlib;files='+repr(r['component_sources_sha256'])+';print(json.dumps({f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files}))';current=json.loads(remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True).stdout);assert current==r['component_sources_sha256']
archive=A/(tag+'.qualification.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for suffix in ['.log','-guard.json','.json','.json.maps.txt','.json.pt']:t.add(A/(tag+suffix),arcname=tag+suffix)
 for name in ['OPS_MATH_ADALN3_CPU_PROOF.json','ops_math_adaln3_layout.py','ops_math_adaln3_support.py','worker_ops_math_adaln3.py']:t.add(P/name,arcname=name)
for f in [archive]:
 dest=N+'/'+f.name;code='from pathlib import Path;import sys;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)';remote('python3 -c '+shlex.quote(code),input=f.read_bytes());assert remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True).stdout.split()[0]==sha(f)
q={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_API_original_reference_inputbits_flush_private_kernels_guard_and_dual_archive_verified':True,'execution_hostname':r['runtime']['hostname'],'tag':tag,'zero_formal_coverage':True,'complete_APIs_per_call':3,'actual_kernel_launch_counts':counts,'actual_private_Mul_Add_objects_sha256':objects,'component_sources_sha256':r['component_sources_sha256'],'formal_source_sha256':{n:sha(P/n) for n in ['ops_math_adaln3_layout.py','ops_math_adaln3_support.py','worker_ops_math_adaln3.py','OPS_MATH_ADALN3_CPU_PROOF.json']},'archive':archive.name,'archive_sha256':sha(archive),'no_kernel_source_or_compiler_flags_changed':True};f=A/'ADALN3_HOST_QUALIFICATION.json';f.write_text(json.dumps(q,indent=2)+'\n');code='from pathlib import Path;import sys;Path('+repr(N+'/'+f.name)+').write_bytes(sys.stdin.buffer.read())';remote('python3 -c '+shlex.quote(code),input=f.read_bytes());assert remote('sha256sum '+N+'/'+f.name,stdout=subprocess.PIPE,text=True).stdout.split()[0]==sha(f);print('ADALN3 qualification passed',counts,sha(archive))
