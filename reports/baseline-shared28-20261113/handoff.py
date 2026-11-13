"""Lossless raw CPU storage handoff with SHA verification and mmap loading."""
import pathlib,hashlib,shutil,fcntl,gzip,time,ctypes,ctypes.util
def compress(raw):
 b=bytes(raw);n=len(b);shuffled=b[::2]+b[1::2]
 z=ctypes.CDLL(ctypes.util.find_library('zstd'));z.ZSTD_compressBound.argtypes=[ctypes.c_size_t];z.ZSTD_compressBound.restype=ctypes.c_size_t
 z.ZSTD_compress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];z.ZSTD_compress.restype=ctypes.c_size_t
 out=ctypes.create_string_buffer(z.ZSTD_compressBound(n));size=z.ZSTD_compress(out,len(out),shuffled,n,1);assert size<=len(out)
 return out.raw[:size]
def decompress(data,n):
 z=ctypes.CDLL(ctypes.util.find_library('zstd'));z.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t];z.ZSTD_decompress.restype=ctypes.c_size_t
 out=ctypes.create_string_buffer(n);size=z.ZSTD_decompress(out,n,data,len(data));assert size==n
 b=out.raw;raw=bytearray(n);split=(n+1)//2;raw[::2]=b[:split];raw[1::2]=b[split:];return raw


def sha(f):
 with open(f,'rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def _save(bundle,root,tid,timeout=120):
 import torch
 root=pathlib.Path(root);dirs=[root/'spool',root/'spool']
 for d in dirs:d.mkdir(exist_ok=True)
 stores={};specs=[];ids={}
 def encode(x):
  if isinstance(x,torch.Tensor):
   assert x.device.type=='cpu' and x.layout==torch.strided
   if id(x) not in ids:
    s=x.untyped_storage();key=(s.data_ptr(),s.nbytes());store=stores.setdefault(key,{'index':len(stores),'storage':s})
    ids[id(x)]=len(specs);specs.append({'storage':store['index'],'dtype':str(x.dtype).split('.')[-1],'shape':list(x.shape),'stride':list(x.stride()),'offset':x.storage_offset()})
   return ['tensor',ids[id(x)]]
  if isinstance(x,dict):return ['dict',[(k,encode(v)) for k,v in x.items()]]
  if isinstance(x,tuple):return ['tuple',[encode(v) for v in x]]
  if isinstance(x,list):return ['list',[encode(v) for v in x]]
  return ['value',x]
 tree=encode(bundle);parts=[];chunk=1<<50
 try:
  for store in stores.values():
   s=store['storage'];raw=torch.empty(0,dtype=torch.uint8).set_(s,0,(s.nbytes(),),(1,));view=memoryview(raw.numpy());files=[]
   for offset in range(0,s.nbytes(),chunk):
    size=min(chunk,s.nbytes()-offset);packed=view[offset:offset+size];deadline=time.monotonic()+timeout
    while True:
     with (root/'handoff-capacity.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX)
      available=[shutil.disk_usage(d).free-reserve for d,reserve in zip(dirs,[51*1024**3,51*1024**3])]
      choice=next((i for i,free in enumerate(available) if free>len(packed)+64*1024**2),None)
      if choice is not None:
       f=dirs[choice]/f'{tid}.shard{store["index"]}-{offset}.raw';parts.append(f)
       f.write_bytes(packed)
       files.append({'path':str(f),'offset':offset,'raw_bytes':size,'size':f.stat().st_size,'sha256':sha(f),'codec':'raw-storage-bytes-v1'});break
     if time.monotonic()>deadline:raise RuntimeError('Raw CPU handoff capacity unavailable; 51GiB disk reserve preserved')
     time.sleep(1)
   store['files']=files;store['raw_bytes']=s.nbytes()
  return {'format':'raw-storage-mmap-v1','tree':tree,'tensors':specs,'storages':[{k:v for k,v in s.items() if k!='storage'} for s in stores.values()]}
 except BaseException:
  for f in parts:f.unlink(missing_ok=True)
  raise

def load(payload):
 import torch
 stores=[]
 for s in payload['storages']:
  if s['files'] and all(f.get('codec')=='raw-storage-bytes-v1' for f in s['files']):
   assert len(s['files'])==1
   f=s['files'][0];path=pathlib.Path(f['path'])
   assert f['offset']==0 and f['raw_bytes']==s['raw_bytes']==f['size']
   assert path.stat().st_size==f['size'] and sha(path)==f['sha256']
   raw=torch.from_file(str(path),shared=False,size=s['raw_bytes'],dtype=torch.uint8)
   path.unlink();stores.append(raw.untyped_storage());continue
  raw=torch.empty(s['raw_bytes'],dtype=torch.uint8);view=memoryview(raw.numpy())
  for f in s['files']:
   path=pathlib.Path(f['path']);assert path.stat().st_size==f['size'] and sha(path)==f['sha256']
   target=view[f['offset']:f['offset']+f['raw_bytes']]
   if f.get('codec')=='zstd1-byte-shuffle2':target[:]=decompress(path.read_bytes(),f['raw_bytes'])
   else:
    with gzip.open(path,'rb') as z:
     n=z.readinto(target);assert n==len(target) and not z.read(1)
   path.unlink()
  stores.append(raw.untyped_storage())
 values=[torch.empty(0,dtype=getattr(torch,s['dtype'])).set_(stores[s['storage']],s['offset'],s['shape'],s['stride']) for s in payload['tensors']]
 def decode(x):
  k,v=x
  if k=='tensor':return values[v]
  if k=='dict':return {a:decode(b) for a,b in v}
  if k=='list':return [decode(a) for a in v]
  if k=='tuple':return tuple(decode(a) for a in v)
  assert k=='value';return v
 return decode(payload['tree'])


def save(bundle,root,tid,timeout=120):
 # Serialize large writers so two partial payloads cannot consume all capacity
 # while neither reaches PREPARED. Small CPU preparation and NPU remain concurrent.
 with (pathlib.Path(root)/'large-handoff.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _save(bundle,root,tid,timeout)
