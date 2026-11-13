"""Bounded, synchronous audit; recording errors abort admission/monitoring."""
import base64
import hashlib
import json
import os
from pathlib import Path
import time


class Audit:
    def __init__(self, path, limit=8*1024*1024):
        self.file=Path(path).open('xb')
        self.size=0
        self.limit=limit
        self.failed=False
    def record(self, kind, **fields):
        if self.failed:raise RuntimeError('audit permanently failed')
        try:
            raw=(json.dumps({'kind':kind,'at':time.monotonic(),**fields},allow_nan=False)+'\n').encode()
            if self.size+len(raw)>self.limit:raise ValueError('audit size limit')
            if self.file.write(raw)!=len(raw):raise OSError('short audit write')
            self.file.flush();os.fsync(self.file.fileno());self.size+=len(raw)
        except BaseException:
            self.failed=True
            raise
    def frame(self, data):
        self.record('raw_frame',bytes_base64=base64.b64encode(data).decode(),length=len(data))
    def close(self):
        self.file.close()


def live_identity(pid,ticks,expected):
    if ticks(pid)!=expected:raise ValueError('worker recycled before maps')
    maps=Path(f'/proc/{pid}/maps').read_bytes()
    if len(maps)>256*1024:raise ValueError('maps cap')
    libraries={}
    for line in maps.decode().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)<6 or not parts[5].startswith('/'):continue
        path=parts[5]
        if not any(x in Path(path).name for x in ['libdrvdsmi','libascend_hal','liburma','libummu','libc.so','libstdc++']):continue
        if path in libraries:continue
        p=Path(path)
        with p.open('rb') as f:
            before=os.fstat(f.fileno())
            h=hashlib.sha256()
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            after=os.fstat(f.fileno())
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('mapped file changed during hash')
        # File identity is the named mapped file, not a hash of resident pages.
        if int(parts[4])!=before.st_ino:raise ValueError('mapped pathname inode changed')
        libraries[path]={'sha256':h.hexdigest(),'inode':before.st_ino,'device':before.st_dev,'size':before.st_size}
    if ticks(pid)!=expected:raise ValueError('worker recycled after maps')
    status=Path(f'/proc/{pid}/status').read_text()
    if len(status)>16384:raise ValueError('worker status cap')
    return {'status':status,'pid':pid,'start_ticks':expected,'maps':maps.decode(),'selected_mapped_files':libraries,
            'limit':'hashes named mapped files, not resident memory; query state not inferred'}
