from pathlib import Path
import json,hashlib,shutil,os,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-state-original-v1');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def capture(work,destination):
 import torch
 out=Path(destination);out.mkdir(parents=True,exist_ok=False);m=json.loads((N/'FORMAL_BUILD_IDENTITY.json').read_text());lib=N/'liboriginal_state_passing.so';obj=N/'original_transformer_state.o';assert sha(lib)==m['binary_sha256'] and sha(obj)==m['object_sha256']
 maps=Path('/proc/self/maps').read_text();assert str(lib) in maps;table=torch._C._dispatch_dump_table('npu_ops_transformer_ext::mambav2_chunk_state_passing');owner=str(N/'source/experimental/mamba/mamba2_chunk_state_passing/torch_interface.cpp');assert any(line.startswith('PrivateUse1:') and owner in line for line in table.splitlines());files={}
 paths=[N/'FORMAL_BUILD_IDENTITY.json',N/'BUILD_RECIPE_RESULT.json',N/'build_transformer_state_original.py',lib,obj,*[N/x for x in m['original_source_sha256']],*[Path(x) for x in m['device_ELFs']]]
 for p in paths:
  dst=out/p.relative_to(N);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);files[str(dst.relative_to(out))]={'sha256':sha(dst),'bytes':dst.stat().st_size}
 for p,v in m['device_ELFs'].items():assert sha(Path(p))==v['sha256']
 (out/'loaded-maps.txt').write_text(maps);(out/'dispatch-table.txt').write_text(table);record={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_CPP_source_and_object_loadedso_deviceELFs_verified':True,'private_dispatch_owner':owner,'kernel_symbol':m['kernel_symbol'],'files':files,'loaded_library_sha256':sha(lib),'device_ELF_sha256':[r['sha256'] for r in m['device_ELFs'].values()],'outside_timed_windows':True,'no_source_kernel_precision_changes':True};(out/'RETENTION.json').write_text(json.dumps(record,indent=2)+'\n');return record
