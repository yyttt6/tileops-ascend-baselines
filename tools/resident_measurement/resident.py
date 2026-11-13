"""CPU-review candidate ONLY: not installed, no guard entrypoint or device CLI.

Parent owns one direct worker. No restart/cache. Production limits are fixed10s.
Original guard run/execute/foreign_processes remain unchanged and NOT wired here.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time
import uuid

VERSION = 'resident-probe-candidate04'
PIN = '1c0e16cb3725a4cebf663b5d2f0339070939a562d4cd2c59158b42f8d465548d'
LIMIT = 65536


def original():
    path = Path(__file__).with_name('original_guard.py')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != PIN:
        raise ValueError('original source pin mismatch')
    spec = importlib.util.spec_from_file_location('frozen_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # importing does not load driver
    return module, data.decode()


def retained_sampler():
    import candidate_guard as guard
    devices = sorted(str(x) for x in Path("/dev").glob("davinci[0-9]*"))
    if devices != ["/dev/davinci1"]:
        raise RuntimeError(f"unexpected device nodes before library load: {devices}")
    lib = guard.load_driver()
    return lambda: guard.query(lib)


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def send(fd, data):
    payload = (json.dumps(data, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if len(payload) > LIMIT:
        raise ValueError('oversized frame')
    # Small requests, exclusive parent writer; partial response writes handled.
    while payload:
        n = os.write(fd, payload)
        payload = payload[n:]


def worker(read_fd, write_fd, session, factory):
    """Dedicated channel: driver stdout/stderr cannot masquerade as a response.

    factory is explicit: retained_sampler for future reviewed device use; CPU
    tests supply a fake. No environment-controlled fake in a deployed guard.
    """
    sample = factory()
    ident = {'pid': os.getpid(), 'ticks': ticks(os.getpid()), 'pgid': os.getpgrp()}
    send(write_fd, {'version': VERSION, 'session': session, 'kind': 'ready', **ident})
    seq = 0
    with os.fdopen(read_fd, 'rb') as stream:
        while True:
            line = stream.readline(LIMIT + 1)
            if not line:
                return
            if len(line) > LIMIT or not line.endswith(b'\n'):
                raise ValueError('request frame')
            req = json.loads(line)
            seq += 1
            if req != {'session': session, 'id': seq}:
                raise ValueError('request sequence/session')
            start = time.monotonic()
            value = sample()
            end = time.monotonic()
            send(write_fd, {'version': VERSION, 'session': session, 'kind': 'sample',
                            **ident, 'id': seq, 'start': start, 'end': end, 'sample': value})


class Client:
    def __init__(self, command, log, *, _cpu_test_timeout=None, reservation_fd=None, checkpoint=lambda: None, audit=None, reject_sample=None, max_requests=64):
        self.audit = audit
        self.reject_sample = reject_sample
        self.max_requests = max_requests
        self.checkpoint = checkpoint
        self.timeout = 10.0 if _cpu_test_timeout is None else _cpu_test_timeout
        self.extra_fds = () if reservation_fd is None else (reservation_fd,)
        self.invalid = False
        self.seq = 0
        self.session = uuid.uuid4().hex
        self.buffer = b''
        self.cleanup = None
        read_req, self.write_req = os.pipe()
        self.read_resp, write_resp = os.pipe()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.read_resp, selectors.EVENT_READ)
        os.set_blocking(self.write_req, False)
        started = time.monotonic()
        try:
            if self.audit:self.audit.record('startup_begin',session=self.session,deadline=started+self.timeout)
            self.process = subprocess.Popen(
                [*command, str(read_req), str(write_resp), self.session],
                pass_fds=(read_req, write_resp, *self.extra_fds), stdout=log, stderr=log,
                start_new_session=False)
        except BaseException:
            for fd in (read_req, self.write_req, self.read_resp, write_resp):
                os.close(fd)
            self.selector.close()
            raise
        finally:
            # Child endpoints no longer belong to parent after successful spawn.
            if hasattr(self, 'process'):
                os.close(read_req)
                os.close(write_resp)
        try:
            self.identity = {'pid': self.process.pid, 'ticks': ticks(self.process.pid), 'pgid': os.getpgrp()}
            self.checkpoint()
            reply = self.receive(started + self.timeout)
            self.check_common(reply, 'ready')
            if self.audit:
                from audit import live_identity
                self.audit.record('ready_accepted',frame=reply,deadline=started+self.timeout)
                self.audit.record('live_worker_identity',**live_identity(self.process.pid,ticks,self.identity['ticks']))
            if time.monotonic()>started+self.timeout:raise TimeoutError('startup audit deadline')
        except BaseException as exc:
            if self.audit and not self.audit.failed:
                try:self.audit.record('startup_rejected',reason=repr(exc))
                except Exception:pass
            try:
                self.fail()
            except BaseException as cleanup_error:
                exc.cleanup_error = repr(cleanup_error)
            exc.worker_cleanup = self.cleanup
            raise

    def check_common(self, reply, kind):
        if (self.process.poll() is not None or ticks(self.process.pid) != self.identity['ticks']
                or os.getpgid(self.process.pid) != self.identity['pgid']
                or reply.get('version') != VERSION or reply.get('session') != self.session
                or reply.get('kind') != kind
                or any(reply.get(k) != v for k, v in self.identity.items())):
            raise ValueError('worker identity/session/version/death')

    def send_request(self, data, deadline):
        payload = (json.dumps(data) + '\n').encode()
        with selectors.DefaultSelector() as writable:
            writable.register(self.write_req, selectors.EVENT_WRITE)
            while payload:
                self.checkpoint()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('IPC send deadline')
                if not writable.select(min(left, .05)):
                    continue
                try:
                    n = os.write(self.write_req, payload)
                except BlockingIOError:
                    continue
                payload = payload[n:]

    def receive(self, deadline):
        while b'\n' not in self.buffer:
            self.checkpoint()
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('worker deadline')
            if not self.selector.select(min(left, .05)):
                continue
            data = os.read(self.read_resp, LIMIT + 1)
            if not data:
                raise ValueError('worker EOF/partial frame')
            if self.audit:self.audit.frame(data)
            self.buffer += data
            if len(self.buffer) > LIMIT:
                raise ValueError('oversized response')
        line, self.buffer = self.buffer.split(b'\n', 1)
        if self.buffer:
            raise ValueError('unsolicited/duplicate response')
        if time.monotonic() > deadline:
            raise TimeoutError('late response')
        value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value, dict):
            raise ValueError('non-object frame')
        return value

    def query(self):
        if self.invalid:
            raise RuntimeError('worker permanently invalid; no restart')
        try:
            self.checkpoint()
            # Bytes outside an outstanding request invalidate the session.
            if self.selector.select(0):
                raise ValueError('unsolicited bytes or worker EOF')
            start = time.monotonic()
            self.seq += 1
            if self.seq>self.max_requests:raise ValueError('request cap')
            if self.audit:self.audit.record('request_send_begin',session=self.session,id=self.seq,identity=self.identity,deadline=start+self.timeout)
            self.send_request({'session': self.session, 'id': self.seq}, start + self.timeout)
            if self.audit:self.audit.record('request_send_end',session=self.session,id=self.seq)
            reply = self.receive(start + self.timeout)
            if self.audit:
                self.audit.record('response_received',session=self.session,id=self.seq,frame=reply)
                status=Path(f'/proc/{self.process.pid}/status').read_text()
                if len(status)>16384:raise ValueError('worker status cap')
                self.audit.record('worker_resource_snapshot',id=self.seq,status=status)
            self.check_common(reply, 'sample')
            a, b = reply.get('start'), reply.get('end')
            if (reply.get('id') != self.seq or type(a) not in (int, float)
                    or type(b) not in (int, float) or not math.isfinite(a)
                    or not math.isfinite(b) or not start <= a <= b <= time.monotonic()):
                raise ValueError('sequence or freshness')
            sample = reply['sample']
            self.validate_sample(sample)
            guard, _ = original()
            if (sample.get('state'), sample.get('reasons')) != guard.classify(sample):
                raise ValueError('classification differs from frozen implementation')
            if self.reject_sample and self.reject_sample(sample):
                raise ValueError('explicit parent transport rejection after saved real response')
            if self.audit:self.audit.record('response_validation_passed',session=self.session,id=self.seq,deadline=start+self.timeout)
            if time.monotonic()>start+self.timeout:raise TimeoutError('response audit deadline')
            # Preserve original failure sample, but retire this worker forever.
            # Caller must feed it to unchanged guard stop logic before any work.
            if (sample['state'] == 'unknown' or sample['errors']
                    or sample['unreadable_proc_pids'] or sample['pid_query_rc'] != 0
                    or sample['health'] != {'rc': 0, 'value': 0}):
                self.fail()
            return sample
        except BaseException as exc:
            if self.audit and not self.audit.failed:
                try:self.audit.record('response_rejected',session=self.session,id=self.seq,reason=repr(exc))
                except Exception:pass
            self.fail()
            raise

    @staticmethod
    def validate_sample(s):
        if not isinstance(s, dict):
            raise ValueError('invalid sample')
        required = {'time','device','errors','utilization','pid_query_rc','pids','health',
                    'memory','core_frequency','power','fd_candidates','unreadable_proc_pids',
                    'observation_protocol','fd_scan_performed','observation_limit','state','reasons'}
        if not required <= s.keys():
            raise ValueError('incomplete sample')
        if s['device'] != 0 or s['fd_scan_performed'] is not False:
            raise ValueError('device/protocol')
        if s['observation_protocol'] != 'reservation-driver-v1-diagnostic04':
            raise ValueError('original protocol changed')
        if set(s['utilization']) != {'aicore','hbm_bw','vector','npu','cube'}:
            raise ValueError('utilization missing')
        for x in [s['health'], *s['utilization'].values()]:
            if not isinstance(x, dict) or type(x.get('rc')) is not int or type(x.get('value')) is not int:
                raise ValueError('invalid driver field')
        for k in ['pids','errors','fd_candidates','unreadable_proc_pids','reasons','core_frequency']:
            if not isinstance(s[k], list):
                raise ValueError('invalid list field')
        if type(s['pid_query_rc']) is not int:
            raise ValueError('invalid pid query rc')
        for p in s['pids']:
            if not isinstance(p, dict) or type(p.get('pid')) is not int or p['pid'] <= 0:
                raise ValueError('invalid process identity')
            if type(p.get('start_ticks')) is not int or type(p.get('uid')) is not int:
                raise ValueError('unresolved process identity')

    def fail(self):
        self.invalid = True
        self.close()

    def close(self):
        if self.cleanup is not None:
            return
        self.invalid = True
        errors = []
        cleanup_started = time.monotonic()
        # Continue cleanup after individual signal/wait/FD errors; receipt is mandatory.
        for action in ('terminate', 'kill'):
            if self.process.poll() is not None:
                break
            try:
                getattr(self.process, action)()
            except ProcessLookupError:
                pass  # Exit race; wait still reaps the child.
            except OSError as exc:
                errors.append({'action': action, 'error': repr(exc)})
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            except OSError as exc:
                errors.append({'action': 'wait', 'error': repr(exc)})
        for action in (self.selector.close, lambda: os.close(self.read_resp),
                       lambda: os.close(self.write_req)):
            try:
                action()
            except OSError as exc:
                errors.append({'action': 'close_ipc', 'error': repr(exc)})
        try:
            observed = {'ticks': ticks(self.process.pid), 'pgid': os.getpgid(self.process.pid)}
        except (FileNotFoundError, ProcessLookupError):
            observed = None
        except OSError as exc:
            observed = {'error': repr(exc)}
            errors.append({'action': 'post_cleanup_identity', 'error': repr(exc)})
        self.cleanup = {'reaped': self.process.poll() is not None, 'errors': errors,
                        'identity': getattr(self, 'identity', None),
                        'started': cleanup_started, 'finished': time.monotonic(),
                        'post_cleanup_proc': observed}
        if not self.cleanup['reaped']:
            errors.append('own worker not reaped; possible uninterruptible sleep')
        if errors:
            raise RuntimeError(str(self.cleanup))
