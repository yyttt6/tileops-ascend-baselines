"""CPU-only log cursor seeding; never reread unchanged CANN log prefixes."""
from pathlib import Path


def seed_capture(directory, previous):
    state={}
    for path in Path(directory).rglob('*.log'):
        try:
            stat=path.stat();key=(str(path),stat.st_dev,stat.st_ino)
            offset,number=previous.get(key,(0,0))
            if stat.st_size<offset:offset,number=0,0
            complete=offset;position=offset
            with path.open('rb') as stream:
                stream.seek(offset)
                while position<stat.st_size:
                    chunk=stream.read(min(1024*1024,stat.st_size-position))
                    if not chunk:break
                    number+=chunk.count(b'\n');last=chunk.rfind(b'\n')
                    if last>=0:complete=position+last+1
                    position+=len(chunk)
            # Keep a partial trailing line unconsumed, exactly like the
            # original capture. Launch attribution filters by event time.
            state[key]=(complete,number)
        except FileNotFoundError:
            # The original collector will discover the replacement inode.
            continue
    return state
