"""CPU-only registry import probe: device imports, DSOs and child execution denied."""
import pathlib,sys,os,json,importlib,traceback,builtins
P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
name=sys.argv[1];assert name in ('ops_nn','ops_math','ops_transformer')
sys.path[:0]=[str(T),str(T/'src')];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
blocked=[]
class DeviceBoundary(RuntimeError):pass
real_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','torch_npu','acl','pyacl','triton','sgl_kernel_npu','vllm_ascend'):
  blocked.append(name);raise DeviceBoundary('runtime import denied during metadata-only probe: '+name)
 return real_import(name,*args,**kwargs)
def audit(event,args):
 if event in ('subprocess.Popen','os.system','os.fork','os.posix_spawn','os.exec'):
  raise DeviceBoundary('child execution denied: '+event)
 if event=='ctypes.dlopen' and args and args[0] is not None:raise DeviceBoundary('DSO load denied in metadata-only probe')
 if event=='import' and len(args)>1 and isinstance(args[1],str) and args[1].endswith('.so') and '/lib-dynload/' not in args[1]:
  raise DeviceBoundary('non-stdlib compiled module denied: '+args[1])
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=os.fsdecode(args[0])
  if path.startswith(('/dev/davinci','/dev/devmm','/dev/hisi_hdc','/dev/ascend')):raise DeviceBoundary('device open denied')
builtins.__import__=guarded_import;sys.addaudithook(audit)
out={'library':name,'metadata_only':True,'no_NPU_imported':True,'blocked_runtime_imports':blocked}
try:
 module=importlib.import_module('benchmarks.ascend_all_pairs.'+name);out.update(status='registry_import_passed',provider_names=sorted(getattr(module,'PROVIDERS',{})))
except BaseException as exc:
 chain=[];cur=exc;seen=set();missing=None
 while cur is not None and id(cur) not in seen:
  seen.add(id(cur));chain.append({'type':type(cur).__name__,'message':str(cur)})
  if isinstance(cur,FileNotFoundError):missing=cur.filename
  cur=cur.__cause__ or cur.__context__
 out.update(status='metadata_import_blocked',exceptions=chain,missing_file=missing,frames=[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(exc.__traceback__)[-5:]])
assert not any(x.startswith(('torch','torch_npu','triton')) for x in sys.modules)
print(json.dumps(out,ensure_ascii=False))
