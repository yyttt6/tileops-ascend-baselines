"""Read-only stdlib CPU evidence; no imports, builds, device access or acceptance."""
from pathlib import Path
import subprocess,json,hashlib,datetime,shlex,re,struct
A=Path(__file__).resolve().parent
remote=r'''from pathlib import Path
import json,hashlib,subprocess
n=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-layernorm-v1');s=n/'source';m=json.loads((n/'LAYERNORM_NATIVE_PACKAGE.v2.json').read_text());root=Path(m['staging_root']);files={}
for rel in ['norm/layer_norm_v4/op_host/layer_norm_v4_tiling.h','norm/layer_norm_v4/op_kernel/layer_norm_v4_apt.h','norm/layer_norm_v4/op_kernel/layer_norm_v4_apt.cpp','norm/layer_norm_v4/op_kernel/arch35/layer_norm_v4_two_pass_perf.h']:
 p=s/rel;b=p.read_bytes();files[rel]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
libs=[]
for p in list((s/'build-server2-v1').rglob('libcust_opmaster_rt2.0.so'))+[root/'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so']:
 if p.is_file():
  b=p.read_bytes();r=subprocess.run(['readelf','-n',str(p)],capture_output=True,text=True);libs.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'elf_notes':r.stdout})
conf=root/'op_impl/ai_core/tbe/kernel/config/ascend950/layer_norm_v4_apt.json';print(json.dumps({'files':files,'tiling_libraries':libs,'config':{'path':str(conf),'sha256':hashlib.sha256(conf.read_bytes()).hexdigest(),'text':conf.read_text()},'observation':'stdlib files/readelf only; no ELF loading or device queries'}))
'''
r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8','openlibing-DevEnv-055409','python3 -I -S -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr;data=json.loads(r.stdout)
log=A/'server1-adaln-decode-ln-v2-040700332862.log';text=log.read_text();lines=[l for l in text.splitlines() if 'LayerNormV4' in l and ('PrintTilingData' in l or 'tilingKey: 500' in l)]
blob=next(l for l in lines if 'PrintTilingData' in l);words=[int(x) for x in blob.split('data(uint32_t):')[1].split()];raw=struct.pack('<'+'I'*len(words),*words)
fields=['a','aBlockFactor','aUbFactor','aUbFactorAlignB32','r','rAlign','formerBlockUbLoops','tailBlockUbLoops','powerOfTwoForR','epsilon','nullptrGamma','nullptrBeta'];values=struct.unpack_from('<qq7ifbb',raw)
assert len(fields)==len(values)
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'CPU_ABI_layout_review_completed_root_cause_not_established','not_measurement_or_coverage':True,'source_evidence':data,'failed_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'selected_tiling_key':500,'host_tiling_blob':dict(zip(fields,values)),'saved_log_lines':lines[:3],'findings':['Key500 is explicitly implemented by original arch35 RegBaseTwoPassPerf; binary filename _100 is not proof of key100 mismatch.','Saved blob decodes under original host schema: a1/r4096, gamma/beta present, epsilon matches frozen input. No field-layout mismatch established.','Kernel signature includes normalized_shape before gamma/beta, matching saved host args; no missing argument established.','Unstripped build vs packaged library SHA differences alone do not prove wrong build. ELF notes retained for further source/build identity review.','Stock-selected earlier run passed; private-selected run failed full frozen checker. Exact fault still unknown; no same-condition native retry or tolerance/kernel change authorized by these CPU findings.'],'next':'Audit package extraction/stripping identity and generated device metadata against actual build receipts before any changed-condition probe.'}
p=A/'LAYERNORM_DECODE_CPU_ABI_DIAGNOSIS.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'record':str(p),'status':report['status'],'decoded':report['host_tiling_blob'],'libs':data['tiling_libraries']}))
