from pathlib import Path
import json,sys,os,traceback,hashlib,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-indices-v1');owner=sys.argv[1];dest=N/sys.argv[2];g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['owner']==owner and g['phase']=='running' and g['child']['pid']==os.getpgrp();r={'status':'initializing','zero_native_or_compiler_calls':True,'not_acceptance':True}
try:
 import asc_op_compile_base.asc_op_compiler.get_op_tiling as t
 t.load_lib();r['normal_compiler_loader_called']=True;r['compiler_python']=t.__file__;r['compiler_python_sha256']=hashlib.sha256(Path(t.__file__).read_bytes()).hexdigest();r['before']={op:t.get_tiling_def(op) is not None for op in ['ArgMaxWithValue']};r['private_load_return']=t.load_custom_lib(str(N/'build-host-v2/libcust_opmaster_rt2.0.so'));r['after']={op:t.get_tiling_def(op) is not None for op in ['ArgMaxWithValue']};r['status']='metadata_query_completed'
except BaseException as e:r.update(status='metadata_query_failed',error=repr(e),traceback=traceback.format_exc())
finally:
 maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps')).write_text(maps);r['loaded_registry_libraries']={l.split()[-1]:hashlib.sha256(Path(l.split()[-1]).read_bytes()).hexdigest() for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and Path(l.split()[-1]).is_file() and any(x in l.split()[-1] for x in ['libregister','libcust_opmaster','liboptiling','libascendcl'])};r['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();dest.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
