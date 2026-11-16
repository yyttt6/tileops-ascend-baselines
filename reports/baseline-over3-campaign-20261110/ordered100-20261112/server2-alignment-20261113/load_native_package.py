"""Guard-only private native DSO load/owner probe; no shared installation."""
import pathlib,os,json,hashlib,ctypes,fcntl,sys,datetime
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1');expected_owner=sys.argv[1];output=N/sys.argv[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ticks(pid):return int(pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
with pathlib.Path('/run/lock/ascend-npu/device-0.lock').open() as lock:
 g=json.load(lock);child=g['child'];assert g['phase']=='running' and g['owner']==expected_owner and child['pid']==os.getpgrp() and ticks(child['pid'])==child['start_ticks'] and ticks(g['pid'])==g['start_ticks'] and g['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Guard kernel lock not held')
m=json.loads((N/'ARTIFACTS-server2-aligned-native-162714998720.json').read_text());assert not m['source_changes'] and m['source_tracked_files_verified']==25610;root=pathlib.Path(m['staging_root']);out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guard_owner':expected_owner,'manifest_sha256':sha(N/'ARTIFACTS-server2-aligned-native-162714998720.json'),'results':[],'not_numeric_or_performance_acceptance':True};handles=[]
def save():output.write_text(json.dumps(out,indent=2)+'\n')
class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int
order=['op_api/lib/libcust_opapi.so','op_proto/es/lib/linux/aarch64/libes_nn.so','op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so','op_impl/ai_core/tbe/op_tiling/liboptiling.so']
for name in order:
 p=root/name;assert sha(p)==m['libraries'][name]['sha256'];row={'path':str(p),'sha256':sha(p),'status':'attempting'};out['results'].append(row);save()
 try:
  lib=ctypes.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);handles.append(lib);row['status']='loaded'
  if name.startswith('op_api/'):
   row['owned_symbols']={}
   for sym in ['aclnnForeachMulList','aclnnForeachMulListGetWorkspaceSize']:
    info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,sym),ctypes.c_void_p),ctypes.byref(info)) and pathlib.Path(info.fname.decode()).resolve()==p.resolve();row['owned_symbols'][sym]=info.fname.decode()
 except OSError as e:row.update(status='load_failed',error=str(e))
 save()
maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);libraries={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in libraries if pathlib.Path(s).is_file()};out['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps({'results':out['results'],'output':str(output)}));assert all(r['status']=='loaded' for r in out['results']),'Some native components not loadable under canonical runtime; no ABI symbols patched'
