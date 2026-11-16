"""Offline provenance/launch audit and SHA-identical dual archive, no NPU access."""
from pathlib import Path
import sys,json,hashlib,subprocess,shlex,tarfile,datetime
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';host=sys.argv[1];assert host in ['H1','H2']
state_name='ORIGINAL_MATH_NORM_H2_QUALIFICATION.json' if host=='H2' else 'SERVER1_ORIGINAL_MATH_NORM_QUALIFICATION.json'
s=json.loads((A/state_name).read_text());tag=s['tag'];N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1';H='openlibing-DevEnv-055409' if host=='H2' else 'openlibing-DevEnv-661495'
def remote(cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',H,cmd],check=True,**kw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
g=json.loads((A/(tag+'-guard.json')).read_text());r=json.loads((A/(tag+'.json')).read_text());build=json.loads((A/('ORIGINAL_MATH_LAYERNORM_'+host+'_CPU_BUILD.json')).read_text());cpu=json.loads((D/'ORIGINAL_MATH_NORM_CPU_PROOF.json').read_text());align=json.loads((A/'ORIGINAL_MATH_LAYERNORM_CPU_BUILD_ALIGNMENT.json').read_text())
assert g['phase']=='finished' and g['workload_exit_code']==0 and g['performance_valid'] is True
assert len(r['checks'])==4 and r['full_output_and_inputs_verified'] and r['original_1GiB_flush_verified'] and r['loaded_original_private_binary_present']
assert r['loaded_original_binary_sha256']==build['binary_sha256'] and len(cpu['checks'])==8 and cpu['no_NPU_imported']
assert align['original_pin_source_gitblob_compiler_CXX_Torch_ABI_and_normalized_commands_equal']
launch=[]
with (A/(tag+'.log')).open(errors='replace') as f:
 for no,line in enumerate(f,1):
  if 'StreamLaunchKernelV2:' in line and 'layer_norm_fp32' in line:launch.append({'line':no,'text':line.strip()})
assert len(launch)>=4,'Metadata registration alone is not a kernel launch'
files={}
for suffix in ['.log','-guard.json','.json','.json.maps.txt','.json.pt']:
 p=A/(tag+suffix);assert p.is_file();data=p.read_bytes();dest=N+'/'+p.name
 code='from pathlib import Path;import sys;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)'
 remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=data)
 assert remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True).stdout.split()[0]==sha(p);files[p.name]=sha(p)
archive=A/(tag+'.qualification.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for name in files:tar.add(A/name,arcname=name)
 for p in [A/('ORIGINAL_MATH_LAYERNORM_'+host+'_CPU_BUILD.json'),A/'ORIGINAL_MATH_LAYERNORM_CPU_BUILD_ALIGNMENT.json',D/'ORIGINAL_MATH_NORM_CPU_PROOF.json',D/'ops_math_original_norm_layout.py',A/'build_original_math_layernorm_cpu.py',A/'probe_original_math_norm.py']:tar.add(p,arcname=p.name)
code='from pathlib import Path;import sys;p=Path('+repr(N+'/'+archive.name)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)';remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=archive.read_bytes());assert remote('sha256sum '+N+'/'+archive.name,stdout=subprocess.PIPE,text=True).stdout.split()[0]==sha(archive)
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'tag':tag,'full_API_embedded_kernel_source_build_guard_and_dual_archive_verified':True,'zero_formal_coverage':True,'build_record':build,'kernel_launch_evidence':launch,'files_sha256':files,'archive':archive.name,'archive_sha256':sha(archive),'remote_archive':N+'/'+archive.name,'no_binary_crosshost_byte_equality_claim':True};put(A/('ORIGINAL_MATH_NORM_'+host+'_IDENTITY_AUDIT.json'),out);print(host,'qualified',len(launch),'launches',archive.name,sha(archive))
