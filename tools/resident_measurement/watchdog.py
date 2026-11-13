"""Bounded owner-only process-tree watchdog; never inspects device FDs.

Guard/worker group and foreground group both tracked by PID/startticks/PGID.
No kill of unrelated jobs. Existing guard handles normal foreground cleanup.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    try:
        f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'ticks':int(f[19]),'pgid':int(f[2]),'state':f[0]}
    except (FileNotFoundError,ProcessLookupError):return None


def observe(owned):
    for p in list(owned.values()):
        current=identity(p['pid'])
        if not current or current['ticks']!=p['ticks']:continue
        try:
            # All threads, but only of already-owned processes; no hostwide scan.
            for task in Path(f"/proc/{p['pid']}/task").iterdir():
                for child in (task/'children').read_text().split():
                    item=identity(int(child))
                    if item:owned[(item['pid'],item['ticks'])]=item
        except (FileNotFoundError,ProcessLookupError):pass


def alive(owned):
    result=[]
    for p in owned.values():
        current=identity(p['pid'])
        if current and current['ticks']==p['ticks'] and current['state']!='Z':result.append(current)
    return result


def supervise(command, output, timeout=90, grace=5, reap=5):
    output.mkdir(parents=True,exist_ok=False)
    received=[]; old={};events=[];owned={};errors=[]
    def handler(number,frame):received.append({'signal':number,'at':time.monotonic()})
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:old[sig]=signal.signal(sig,handler)
    # Dedicated process only: adopt this invocation's orphans, including a
    # foreground group created between polls just before guard dies.
    if Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().strip():
        raise RuntimeError('watchdog requires a dedicated process with no existing children')
    libc=ctypes.CDLL(None,use_errno=True)
    old_subreaper=ctypes.c_int()
    if libc.prctl(37,ctypes.byref(old_subreaper),0,0,0) or libc.prctl(36,1,0,0,0):
        raise OSError(ctypes.get_errno(),'subreaper unavailable')
    start=time.monotonic();stop=None;child=None;code=78
    def discover():
        root=identity(os.getpid())
        pending={(root['pid'],root['ticks']):root}
        observe(pending)
        for key,item in pending.items():
            if item['pid']!=os.getpid():owned[key]=item
        observe(owned)
    def reap_adopted():
        for item in list(owned.values()):
            if child and item['pid']==child.pid:continue
            try:os.waitpid(item['pid'],os.WNOHANG)
            except ChildProcessError:pass
    def notify(sig):
        groups={p['pgid'] for p in alive(owned)}
        for group in groups:
            if group==os.getpgrp():
                errors.append('refused own watchdog group');continue
            # Only known live owned identities anchor each group signal.
            try:os.killpg(group,sig);events.append({'pgid':group,'signal':sig,'at':time.monotonic()})
            except ProcessLookupError:pass
            except OSError as exc:errors.append(repr(exc))
    try:
        with (output/'guard.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
            ident=identity(child.pid)
            if ident:owned[(ident['pid'],ident['ticks'])]=ident
            while True:
                discover();reap_adopted()
                state=alive(owned);rc=child.poll();now=time.monotonic()
                if rc is not None and not state:
                    code=rc if stop is None else (128+received[0]['signal'] if received else 124)
                    break
                if stop is None and (received or now-start>=timeout or rc is not None):
                    stop=now;notify(received[0]['signal'] if received else signal.SIGTERM)
                if stop is not None and now-stop>=grace:
                    notify(signal.SIGKILL)
                if stop is not None and now-stop>=grace+reap:
                    code=78;break
                time.sleep(.05)
    except BaseException as exc:
        errors.append(repr(exc));code=78
    finally:
        # Record-only handlers stay installed throughout cleanup/receipt writing.
        discover()
        if alive(owned):
            notify(signal.SIGKILL)
        if child and child.poll() is None:
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:errors.append('guard unreaped')
        cleanup_end=time.monotonic()+1
        while time.monotonic()<cleanup_end:
            discover();reap_adopted()
            if not alive(owned):break
            time.sleep(.02)
        remaining=alive(owned)
        if remaining or errors:code=78
        receipt={'exit_code':code,'signals':received,'events':events,'owned':list(owned.values()),
                 'remaining_live':remaining,'errors':errors,'elapsed':time.monotonic()-start,
                 'child_returncode':child.poll() if child else None}
        try:
            with (output/'watchdog-receipt.json').open('w') as f:
                json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
        finally:
            libc.prctl(36,old_subreaper.value,0,0,0)
            for sig,previous in old.items():signal.signal(sig,previous)
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    raise SystemExit(supervise(cmd,a.output))
