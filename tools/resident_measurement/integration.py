"""Review-only integration API, no real driver CLI. Explicit worker command required."""
import json
import os
import signal
import time
from pathlib import Path
import candidate_guard as guard
from resident import Client, VERSION
from audit import Audit


class ParentSignal(BaseException):
    def __init__(self, number):
        self.number = number


class Lifecycle:
    def __init__(self, command, log, timeout=None, checkpoint=lambda: None, audit_path=None, arm="A", max_requests=64):
        self.command, self.log, self.timeout = command, log, timeout
        self.checkpoint = checkpoint
        self.max_requests = max_requests
        self.audit_path = audit_path
        self.audit = None
        self.arm = arm
        self.injected = False
        self.fd = None
        self.client = None
        self.failed = None
        self.receipt = {'version': VERSION, 'samples': [], 'parent_pid': os.getpid(),
                        'parent_pgid': os.getpgrp(), 'worker': None}

    def locked(self, fd):
        self.fd = fd
        self.receipt['lock_inode'] = os.fstat(fd).st_ino
        self.receipt['lock_acquired_at'] = time.monotonic()

    def reject_sample(self, sample):
        if self.arm!='B' or self.injected:return False
        meta=guard.read_metadata(self.fd) or {}
        child=meta.get('child') or {}
        pid=child.get('pid')
        if (meta.get('phase')!='running' or not pid or sample.get('errors')
                or sample.get('health')!={'rc':0,'value':0}
                or not sample.get('pids')):return False
        try:
            if guard.proc_identity(pid)!=child or os.getpgid(pid)!=pid:return False
            state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
            if state=='Z':return False
            matching=[]
            for item in sample['pids']:
                if guard.proc_identity(item['pid'])==item and os.getpgid(item['pid'])==pid:
                    matching.append(item)
            if not matching:return False
        except (FileNotFoundError,ProcessLookupError):return False
        self.audit.record('synthetic_transport_reject',foreground=child,
                          foreground_pgid=pid,actual_sample_owners=matching,scope='saved real response; not driver error')
        self.injected=True
        return True

    def sample(self):
        if self.failed:
            return {'state': 'unknown', 'reasons': [self.failed]}
        try:
            if self.fd is None:
                raise RuntimeError('worker cannot start outside reservation')
            if self.client is None:
                self.audit = Audit(self.audit_path) if self.audit_path else None
                self.client = Client(self.command, self.log, reservation_fd=self.fd,
                                     _cpu_test_timeout=self.timeout, checkpoint=self.checkpoint, audit=self.audit, reject_sample=self.reject_sample, max_requests=self.max_requests)
                self.receipt['worker'] = self.client.identity
            value = self.client.query()
            self.receipt['samples'].append(value)
            return value
        except Exception as exc:
            if hasattr(exc, 'worker_cleanup'):
                self.receipt['constructor_cleanup'] = exc.worker_cleanup
            self.failed = repr(exc)
            return {'state': 'unknown', 'reasons': [self.failed]}

    def close(self):
        try:
            if self.client:
                self.client.close()
        finally:
            self.receipt['failure'] = self.failed
            self.receipt['cleanup'] = self.client.cleanup if self.client else None
            self.receipt['lock_open_at_cleanup'] = self.fd is not None and self.fd_open()
            self.receipt['cleanup_hook_finished_at'] = time.monotonic()
            self.receipt['arm'] = self.arm
            self.receipt['injection_exercised'] = self.injected
            self.receipt['audit_failed'] = self.audit.failed if self.audit else None
            if self.audit:self.audit.close()

    def fd_open(self):
        try:
            os.fstat(self.fd)
            return True
        except OSError:
            return False


def run_candidate(args, command, log, receipt_path, *, cpu_timeout=None, arm="A", max_requests=64):
    received = []
    def recorded(number, frame):
        # Never raise asynchronously: constructor/fail/close/receipt are protected.
        received.append({'number': number, 'at': time.monotonic()})
    def checkpoint():
        if received:
            raise ParentSignal(received[0]['number'])
    life = Lifecycle(command, log, cpu_timeout, checkpoint, Path(receipt_path).with_name("ipc-audit.jsonl"), arm, max_requests)
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        handlers[sig] = signal.signal(sig, recorded)
    code = guard.UNAVAILABLE
    try:
        code = guard.run(args, sample_fn=life.sample, cleanup_fn=life.close,
                         on_locked=life.locked)
    except ParentSignal as exc:
        life.receipt['constructor_cleanup'] = getattr(exc, 'worker_cleanup', None)
        code = 128 + exc.number
    except Exception as exc:
        life.receipt['outer_error'] = repr(exc)
    finally:
        # Keep record-only handlers installed until receipt is fully written.
        # Original execute temporarily overrides them with its own record-only
        # handlers; neither version throws during cleanup.
        if received:
            code = 128 + received[0]['number']
        if life.receipt.get('cleanup') is None:
            life.receipt['cleanup'] = life.receipt.get('constructor_cleanup')
        cleanup = life.receipt.get('cleanup')
        if cleanup and life.receipt['worker'] is None:
            life.receipt['worker'] = cleanup.get('identity')
        life.receipt['signals'] = received
        life.receipt['exit_code'] = code
        life.receipt['receipt_written_at'] = time.monotonic()
        life.receipt['parent_lock_closed_after_run'] = life.fd is None or not life.fd_open()
        target = Path(receipt_path)
        try:
            with target.open('w') as out:
                json.dump(life.receipt, out, indent=2)
                out.flush()
                os.fsync(out.fileno())
                if received and code != 128 + received[0]['number']:
                    code = 128 + received[0]['number']
                    life.receipt['exit_code'] = code
                    out.seek(0); out.truncate()
                    json.dump(life.receipt, out, indent=2)
                    out.flush(); os.fsync(out.fileno())
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return code
