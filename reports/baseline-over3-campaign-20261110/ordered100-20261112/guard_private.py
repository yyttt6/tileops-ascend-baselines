#!/usr/bin/env python3
"""Host-wide cooperative reservation for this machine's single Ascend NPU.

Only the Python standard library is required. See docs/NPU共享锁.md.
"""

import argparse
import ctypes as C
import datetime as dt
import fcntl
import json
import math
import platform
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import threading

LOCK_DIR = Path('/run/lock/ascend-npu')
LOCK_NAME = 'device-0.lock'
QUEUE_NAME = 'device-0.queue'
DRIVER = '/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so'
SCRIPT = str(Path(__file__).resolve())
PROBE_SCRIPTS = {SCRIPT, '/usr/local/bin/npu-guard', '/root/dyq/workspace/tools/npu_guard.py'}
PROFILE_PATH = Path('/etc/ascend-npu-guard.json')
PROBE_SOCKET = Path('/run/npu-guard/probe.sock')
PROBE_HELPER = Path('/usr/local/libexec/ascend-npu/probe-root')
ABI_PATH = Path('/etc/ascend-npu-guard-abi.json')
BUSY = 75
CONFLICT = 76
UNAVAILABLE = 78


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def device_profile(path=PROFILE_PATH):
    """Read an explicit administrator profile; keep the existing host default.

    A container profile changes device-node mapping and describes the scope of
    its cooperative lock. It does not suppress any physical-device checks.
    """
    default = {'scope': 'host', 'logical_device': 0, 'device_node': '/dev/davinci0'}
    if not path.exists():
        return default
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
            or info.st_mode & 0o022 or info.st_nlink != 1):
        raise ValueError('device profile must be a root-owned, non-writable regular file')
    profile = json.loads(path.read_text())
    if profile.get('scope') not in ('host', 'container') or type(profile.get('logical_device')) is not int or profile['logical_device'] != 0:
        raise ValueError('only an explicit host/container logical-device-0 profile is supported')
    node = profile.get('device_node', '')
    if (not isinstance(node, str) or not node.startswith('/dev/davinci')
            or not node[len('/dev/davinci'):].isdigit()):
        raise ValueError('invalid configured Ascend device node')
    if profile['scope'] == 'container' and not profile.get('authorization'):
        raise ValueError('container profile requires its recorded authorization scope')
    return profile


def proc_identity(pid):
    """PID start ticks prevent confusing a recycled PID with an old owner."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]),
                'uid': Path(f'/proc/{pid}').stat().st_uid, 'pgid': int(fields[2])}
    except (OSError, ValueError, IndexError):
        return {'pid': pid, 'start_ticks': None, 'uid': None}


class ResourcePara(C.Structure):
    _fields_ = [('owner_type', C.c_uint), ('owner_id', C.c_uint),
                ('resource_type', C.c_uint), ('tsid', C.c_uint),
                ('reserve', C.c_uint * 8)]


class ResourceInfo(C.Structure):
    _fields_ = [('buf_len', C.c_uint), ('buf', C.c_void_p),
                ('reserve', C.c_uint * 8)]


class MemoryInfo(C.Structure):
    _fields_ = [('total_kib', C.c_uint), ('used_kib', C.c_uint),
                ('reserve', C.c_char * 38)]


class CoreFrequency(C.Structure):
    _fields_ = [('rated_mhz', C.c_uint), ('current_mhz', C.c_uint)]


class PowerInfo(C.Structure):
    _fields_ = [('power', C.c_ushort)]


def readonly_guard_probe(entry):
    """Known stdlib-only observers do not create compute contexts.

    This is cooperative classification, not an authorization boundary. An
    uncooperative user can bypass flock altogether. Unknown tools stay candidates.
    """
    try:
        args = (entry / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        tail = args[1:]
        if tail[:3] == [b'-I', b'-S', b'-B']:
            tail = tail[3:]
        return (len(tail) == 2 and tail[1] == b'_probe'
                and os.path.realpath(os.fsdecode(tail[0])) in PROBE_SCRIPTS)
    except (OSError, ValueError):
        return False


def proc_fd_identity(entry):
    """Diagnostic evidence only; never waives a visibility failure."""
    observation = {'observed_at': now(), 'pid': int(entry.name)}
    try:
        fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        observation.update(start_ticks=int(fields[19]), state=fields[0],
                           ppid=int(fields[1]), pgid=int(fields[2]))
    except (OSError, ValueError, IndexError) as exc:
        observation.update(start_ticks=None, error_type=type(exc).__name__,
                           errno=getattr(exc, 'errno', None))
    return observation


def device_fd_nodes(entry):
    """Read every FD; container root may need the target's filesystem UID.

    Matching credentials is an actual read, not a PermissionError exemption.
    Mixed-ID or non-dumpable processes still fail closed without SYS_PTRACE.
    Only the disposable, single-caller probe changes its thread credentials.
    """
    def read():
        nodes = set()
        for fd in list((entry / 'fd').iterdir()):
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            if target.startswith(('/dev/davinci', '/dev/devmm', '/dev/hisi_hdc')):
                nodes.add(target)
        return nodes
    try:
        return read()
    except PermissionError:
        if os.geteuid() != 0:
            raise
    ids = {line.split(':', 1)[0]: list(map(int, line.split()[1:]))
           for line in (entry / 'status').read_text().splitlines()
           if line.startswith(('Uid:', 'Gid:'))}
    if len(set(ids['Uid'][:3])) != 1 or len(set(ids['Gid'][:3])) != 1:
        raise PermissionError('mixed-credential process requires host SYS_PTRACE observation')
    libc = C.CDLL(None, use_errno=True)
    for name in ('setfsuid', 'setfsgid'):
        getattr(libc, name).argtypes = [C.c_uint]
        getattr(libc, name).restype = C.c_uint
    old_uid, old_gid = libc.setfsuid(-1), libc.setfsgid(-1)
    try:
        libc.setfsgid(ids['Gid'][0])
        libc.setfsuid(ids['Uid'][0])
        if libc.setfsuid(-1) != ids['Uid'][0] or libc.setfsgid(-1) != ids['Gid'][0]:
            raise PermissionError('cannot assume target filesystem credentials for observation')
        return read()
    finally:
        libc.setfsuid(old_uid)
        libc.setfsgid(old_gid)
        if libc.setfsuid(-1) != old_uid or libc.setfsgid(-1) != old_gid:
            raise RuntimeError('cannot restore observer filesystem credentials')


def scan_device_fds(proc=Path('/proc'), diagnostics=None):
    """Conservatively include runtime initialization, not just /dev/davinci0.

    A manager/HDC FD is a candidate, not proof of compute activity. An
    incomplete host view must never be interpreted as an empty device.
    """
    candidates, unreadable = [], []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        before = proc_fd_identity(entry) if diagnostics is not None else None
        operation, path = 'listdir', entry / 'fd'
        try:
            for attempt in range(3):
                try:
                    nodes = device_fd_nodes(entry)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    # Credential-changing exec/setuid has a short protected
                    # transition. Retry the complete read, never assume empty.
                    time.sleep(.005)
            if nodes and not readonly_guard_probe(entry):
                candidates.append({**proc_identity(int(entry.name)), 'nodes': sorted(nodes)})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            # An exiting process can vanish during procfs's permission check.
            # Never waive a permission denial for a process that still exists.
            if not entry.exists():
                try:
                    os.kill(int(entry.name), 0)
                except ProcessLookupError:
                    continue
                except OSError:
                    pass
            unreadable.append(int(entry.name))
            if diagnostics is not None:
                after = proc_fd_identity(entry)
                diagnostics.append({'pid': int(entry.name), 'operation': operation,
                    'path': str(path), 'errno': exc.errno, 'error_type': type(exc).__name__,
                    'error_message': str(exc),
                    'before': before, 'after': after,
                    'same_start_ticks': before['start_ticks'] is not None and before['start_ticks'] == after['start_ticks'],
                    'decision': 'unreadable_fail_closed'})
    return candidates, unreadable


def classify(sample):
    reasons = []
    if sample.get('pids'):
        reasons.append('driver reports NPU processes')
    if sample.get('fd_candidates'):
        reasons.append('other processes hold Ascend device FDs (possibly initializing)')
    if any(v.get('rc') == 0 and 0 < v.get('value', 0) <= 100
           for v in sample.get('utilization', {}).values()):
        reasons.append('driver reports positive utilization')
    if reasons:
        return 'busy', reasons
    errors = list(sample.get('errors', []))
    if sample.get('unreadable_proc_pids'):
        errors.append('incomplete /proc FD visibility; use a host account with sufficient permissions')
    if sample.get('health') != {'rc': 0, 'value': 0}:
        errors.append('device health is unavailable or not healthy')
    if sample.get('pid_query_rc') != 0:
        errors.append('driver process query failed')
    if errors:
        return 'unknown', errors
    return 'idle', ['no driver processes or Ascend FD candidates in this sample']


def raw_probe():
    """Read-only DSMI calls in a disposable process, using installed ABI headers.

    dsmi_common_interface.h: PROCESS_RESOURCE=2, DEV_PROCESS_PID=8.
    Ascend950 asys handler: get_device_info(MEMORY=9, HBM_MEMORY=35).
    Never initialize torch/ACL, submit work, reset, or alter device settings.
    """
    result = {'time': now(), 'device': 0, 'errors': [], 'utilization': {}}
    try:
        result['abi'] = verify_abi_identity()
        profile = device_profile()
        result['coordination_scope'] = profile['scope']
        result['device_node'] = profile['device_node']
        devices = sorted(str(p) for p in Path('/dev').glob('davinci[0-9]*'))
        if devices != [profile['device_node']]:
            result['errors'].append(f"expected only {profile['device_node']}; found {devices}")
        lib = C.CDLL(DRIVER)
        if profile['scope'] == 'container':
            lib.dsmi_get_phyid_from_logicid.argtypes = [C.c_uint, C.POINTER(C.c_uint)]
            lib.dsmi_get_phyid_from_logicid.restype = C.c_int
            physical = C.c_uint()
            mapping_rc = lib.dsmi_get_phyid_from_logicid(0, C.byref(physical))
            result['logical_to_physical'] = {'rc': mapping_rc, 'logical': 0, 'physical': physical.value}
            expected_physical = int(profile['device_node'][len('/dev/davinci'):])
            if mapping_rc != 0 or physical.value != expected_physical:
                result['errors'].append('driver logical-to-physical mapping differs from the configured device node')
        lib.dsmi_get_resource_info.argtypes = [C.c_uint, C.POINTER(ResourcePara), C.POINTER(ResourceInfo)]
        lib.dsmi_get_device_health.argtypes = [C.c_int, C.POINTER(C.c_uint)]
        lib.dsmi_get_device_utilization_rate.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_uint)]
        lib.dsmi_get_device_info.argtypes = [C.c_uint, C.c_int, C.c_uint, C.c_void_p, C.POINTER(C.c_uint)]
        lib.dsmi_get_aicore_info.argtypes = [C.c_int, C.POINTER(CoreFrequency)]
        lib.dsmi_get_device_power_info.argtypes = [C.c_int, C.POINTER(PowerInfo)]
        for name in ('dsmi_get_resource_info', 'dsmi_get_device_health',
                     'dsmi_get_device_utilization_rate', 'dsmi_get_device_info',
                     'dsmi_get_aicore_info', 'dsmi_get_device_power_info'):
            getattr(lib, name).restype = C.c_int
        pids = (C.c_int * 64)()
        para = ResourcePara(2, 0, 8, 0)
        info = ResourceInfo(C.sizeof(pids), C.cast(pids, C.c_void_p))
        rc = lib.dsmi_get_resource_info(0, C.byref(para), C.byref(info))
        result['pid_query_rc'] = rc
        result['pids'] = [proc_identity(pid) for pid in pids if pid > 0] if rc == 0 else []
        if rc == 0 and (info.buf_len > C.sizeof(pids) or any(pid < 0 for pid in pids)):
            result['errors'].append('invalid driver process response')
        if len(result['pids']) == 64:
            result['errors'].append('process list may be truncated')
        health = C.c_uint()
        rc = lib.dsmi_get_device_health(0, C.byref(health))
        result['health'] = {'rc': rc, 'value': health.value}
        for kind, name in [(2, 'aicore'), (10, 'hbm_bw'), (12, 'vector'), (13, 'npu'), (14, 'cube')]:
            value = C.c_uint()  # Some current driver paths write only the low byte.
            rc = lib.dsmi_get_device_utilization_rate(0, kind, C.byref(value))
            result['utilization'][name] = {'rc': rc, 'value': value.value}
            if rc == 0 and value.value > 100:
                result['errors'].append(f'invalid {name} utilization')
        if result['utilization']['npu']['rc'] != 0:
            result['errors'].append('NPU overall utilization query failed')
        mem = MemoryInfo()
        size = C.c_uint(C.sizeof(mem))
        rc = lib.dsmi_get_device_info(0, 9, 35, C.byref(mem), C.byref(size))
        result['memory'] = {'rc': rc, 'total_mib': mem.total_kib / 1024, 'used_mib': mem.used_kib / 1024}
        result['core_frequency'] = []
        for component in (0, 1):
            freq = CoreFrequency()
            rc = lib.dsmi_get_aicore_info(component << 16, C.byref(freq))
            result['core_frequency'].append({'rc': rc, 'mhz': freq.current_mhz})
        power = PowerInfo()
        rc = lib.dsmi_get_device_power_info(0, C.byref(power))
        result['power'] = {'rc': rc, 'watts': power.power / 10}
    except (OSError, AttributeError, ValueError, TypeError) as exc:
        result['errors'].append(str(exc))
    result['proc_fd_errors'] = []
    result['fd_candidates'], result['unreadable_proc_pids'] = scan_device_fds(
        diagnostics=result['proc_fd_errors'])
    result['state'], result['reasons'] = classify(result)
    return result


def local_probe():
    try:
        completed = subprocess.run([sys.executable, '-I', '-S', '-B', SCRIPT, '_probe'], capture_output=True,
                                   text=True, timeout=10, check=True)
        # Some driver builds print diagnostic messages on stdout, including
        # during library teardown. Keep our structured record separate.
        for line in reversed(completed.stdout.splitlines()):
            try:
                sample = json.loads(line)
            except ValueError:
                continue
            if isinstance(sample, dict) and sample.get('state') in ('idle', 'busy', 'unknown') and 'reasons' in sample:
                return sample
        raise ValueError('probe did not produce a structured status record')
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        detail = {'type': type(exc).__name__, 'message': str(exc)}
        if isinstance(exc, subprocess.CalledProcessError):
            detail['returncode'] = exc.returncode
        for field in ('stdout', 'stderr'):
            value = getattr(exc, field, None)
            if value:
                detail[field + '_tail'] = (value.decode(errors='replace')
                                           if isinstance(value, bytes) else value)[-4096:]
        return {'time': now(), 'state': 'unknown', 'reasons': [f'probe failed: {exc}'],
                'probe_error': detail}


def trusted_root_path(path, *, socket_file=False):
    """No unprivileged caller may choose the privileged observer's code/socket."""
    info = path.lstat()
    good_type = stat.S_ISSOCK(info.st_mode) if socket_file else stat.S_ISREG(info.st_mode)
    if not good_type or info.st_uid != 0 or (not socket_file and info.st_mode & 0o022):
        raise PermissionError(f'untrusted root observer path: {path}')
    for parent in path.parents:
        pi = parent.lstat()
        if not stat.S_ISDIR(pi.st_mode) or pi.st_uid != 0 or pi.st_mode & 0o022:
            raise PermissionError(f'untrusted observer parent: {parent}')


def verify_abi_identity(path=ABI_PATH):
    """An upgrade requires a fresh host-side C layout/prototype verification."""
    trusted_root_path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 1 or manifest.get('machine') != platform.machine()
            or manifest.get('pointer_bytes') != C.sizeof(C.c_void_p)):
        raise ValueError('DSMI ABI qualification is missing or belongs to another machine ABI')
    for name in ('ResourcePara', 'ResourceInfo', 'MemoryInfo', 'CoreFrequency', 'PowerInfo'):
        cls = globals()[name]
        layout = [C.sizeof(cls)] + [getattr(cls, f[0]).offset for f in cls._fields_]
        if manifest.get('layouts', {}).get(name) != layout:
            raise ValueError(f'DSMI ctypes layout changed: {name}; re-run ABI qualification')
    expected_paths = {str(Path(DRIVER).resolve()),
                      '/usr/local/Ascend/driver/include/dsmi_common_interface.h'}
    if not expected_paths <= {r['path'] for r in manifest.get('files', [])}:
        raise ValueError('DSMI ABI qualification lacks the active driver/header identity')
    for record in manifest['files']:
        source = Path(record['path'])
        # Vendor headers may belong to HwHiAiUser. Their identities are pinned
        # by this root-owned manifest; only executable code requires root-only
        # ownership along its complete path. ctime detects owner-side changes.
        if str(source) == str(Path(DRIVER).resolve()):
            trusted_root_path(source)
        current = source.stat()
        if any(getattr(current, k) != value for k, value in record['stat'].items()):
            raise ValueError(f'DSMI ABI input changed: {source}; re-run npu_guard_abi.py')
    return {'verified': True, 'machine': manifest['machine'],
            'manifest': str(path), 'input_identities_unchanged': True}


def receive_line(connection, limit):
    data = bytearray()
    while b'\n' not in data:
        chunk = connection.recv(min(65536, limit + 1 - len(data)))
        if not chunk:
            raise ValueError('observer closed before a complete response')
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError('observer message too large')
    line, rest = bytes(data).split(b'\n', 1)
    if rest:
        raise ValueError('unexpected extra observer message')
    return json.loads(line)


def broker_probe(path=PROBE_SOCKET):
    trusted_root_path(path, socket_file=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(15)
        connection.connect(str(path))
        _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != 0:
            raise PermissionError('observer server is not root')
        connection.sendall(b'{"version":1,"op":"probe"}\n')
        result = receive_line(connection, 1024 * 1024)
    age = time.monotonic() - result['observed_monotonic']
    if result.get('version') != 1 or not 0 <= age <= 15:
        raise ValueError('invalid or stale privileged observation')
    sample = result['sample']
    if sample.get('state') not in ('idle', 'busy', 'unknown') or not isinstance(sample.get('reasons'), list):
        raise ValueError('invalid privileged observation schema')
    sample['observer'] = 'root-unix-service'
    return sample


def privileged_helper_probe():
    # A fixed native launcher normalizes real/effective/saved IDs before exec.
    # No long-lived sudo monitor with mixed IDs obstructs container observers.
    trusted_root_path(PROBE_HELPER)
    result = subprocess.run([str(PROBE_HELPER)],
                            capture_output=True, text=True, timeout=15, check=True,
                            env={'PATH': '/usr/bin:/bin', 'LANG': 'C'})
    for line in reversed(result.stdout.splitlines()):
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        if isinstance(sample, dict) and sample.get('state') in ('idle', 'busy', 'unknown') and 'reasons' in sample:
            sample['observer'] = 'root-fixed-probe-helper'
            return sample
    raise ValueError('privileged helper returned no observation')


def probe():
    try:
        if PROBE_SOCKET.exists():
            return broker_probe(PROBE_SOCKET)
        if os.geteuid() != 0:
            return privileged_helper_probe()
        return local_probe()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return {'time': now(), 'state': 'unknown',
                'reasons': [f'privileged observation unavailable: {exc}'],
                'probe_error': {'type': type(exc).__name__, 'message': str(exc)}}


def serve_probes(path=PROBE_SOCKET, sample_fn=local_probe):
    """Bounded read-only observer. Protocol accepts no commands, paths or PIDs."""
    if os.geteuid() != 0:
        raise PermissionError('only root may serve device observations')
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    trusted_root_path(Path(SCRIPT))
    for parent in (path.parent, *path.parent.parents):
        st = parent.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
            raise PermissionError('observer socket directory is not protected')
    singleton = os.open(path.with_suffix('.service-lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    if not try_lock(singleton):
        os.close(singleton)
        raise OSError('an observer service already owns this socket')
    capacity = threading.BoundedSemaphore(8)
    sampling = threading.Lock()
    cache = {}

    def handle(connection):
        try:
            with connection:
                connection.settimeout(2)
                request = receive_line(connection, 128)
                if request != {'version': 1, 'op': 'probe'}:
                    raise ValueError('only the fixed read-only probe request is accepted')
                with sampling:
                    if time.monotonic() - cache.get('observed_monotonic', -1e9) > .2:
                        sample = sample_fn()
                        cache.update(version=1, observed_monotonic=time.monotonic(), sample=sample)
                    response = (json.dumps(cache) + '\n').encode()
                connection.sendall(response)
        except (OSError, ValueError, TypeError):
            pass  # A malformed client can only lose its own query.
        finally:
            capacity.release()

    try:
        if path.exists():
            trusted_root_path(path, socket_file=True)
            path.unlink()  # Own service singleton held; never touches the NPU lock.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            path.chmod(0o666)
            listener.listen(16)
            while True:
                connection, _ = listener.accept()
                if not capacity.acquire(blocking=False):
                    connection.close()
                    continue
                threading.Thread(target=handle, args=(connection,), daemon=True).start()
    finally:
        os.close(singleton)


def open_lock():
    """Root creates a persistent inode. Clients never unlink or replace it."""
    if not LOCK_DIR.exists():
        if os.geteuid() != 0:
            raise PermissionError('root must run npu_guard.py init once after boot')
        try:
            LOCK_DIR.mkdir(mode=0o755)
        except FileExistsError:
            pass
    directory = LOCK_DIR.lstat()
    if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != 0 or directory.st_mode & 0o022:
        raise PermissionError(f'{LOCK_DIR} must be a root-owned, non-writable real directory')
    path = LOCK_DIR / LOCK_NAME
    if os.geteuid() == 0:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o666)
        except FileExistsError:
            pass
        else:
            os.fchmod(fd, 0o666)
            os.close(fd)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_nlink != 1:
        os.close(fd)
        raise PermissionError('lock must be a root-owned regular file with one link')
    return fd


def try_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def open_queue():
    """The root-owned sticky directory is shared by all cooperating UIDs."""
    path = LOCK_DIR / QUEUE_NAME
    if os.geteuid() == 0:
        try:
            path.mkdir(mode=0o1777)
        except FileExistsError:
            pass
        else:
            path.chmod(0o1777)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o1777:
        os.close(fd)
        raise PermissionError('root must initialize a root-owned 1777 queue directory')
    return fd


class FifoTicket:
    """FIFO among published tickets; the original inode still owns the device.

    A directory flock serializes only queue metadata syscalls, never probes or
    workloads. Nonblocking acquisition keeps --wait bounded even if a publisher
    is stopped. Ticket flock, not PID/age, proves liveness. Old clients retain
    mutual exclusion but cannot participate in FIFO until upgraded.
    """
    def __init__(self):
        self.directory = open_queue()
        self.fd = None
        self.name = None
        self.owner_predecessor = None

    def _entries(self):
        entries = []
        for name in os.listdir(self.directory):
            serial = name.partition('-')[0]
            if name.endswith('.ticket') and serial.isascii() and serial.isdecimal():
                entries.append((int(serial), name))
        return sorted(entries)

    def _remove_dead(self, name):
        """Called under the directory lock; never remove a live publisher."""
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=self.directory)
        except FileNotFoundError:
            return True
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise PermissionError('queue ticket must be a regular file')
            if not try_lock(fd):
                return False
            # Sticky directories intentionally forbid deleting another UID's
            # dead ticket. Ignore it now; its owner/root can reclaim it later.
            try:
                os.unlink(name, dir_fd=self.directory)
            except (FileNotFoundError, PermissionError):
                pass
            return True
        finally:
            os.close(fd)

    def _same_owner_live_ticket(self, metadata):
        # Called with the queue-directory flock held. A duplicate command waits
        # without publishing another FIFO ticket; its normal --wait deadline
        # remains in force. Never infer ticket liveness from PID or timestamps.
        for _, name in self._entries():
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=self.directory)
            except FileNotFoundError:
                continue
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise PermissionError('queue ticket must be a regular file')
                if try_lock(fd):
                    continue  # Dead inode: closing this FD releases our probe.
                other = read_metadata(fd)
                if (info.st_uid == os.geteuid() and isinstance(other, dict)
                        and other.get('owner') == metadata['owner']):
                    return name
            finally:
                os.close(fd)
        return None

    def _publish(self, metadata):
        # All publishers hold the directory flock before creating a staging
        # file. Cleanup uses the same mutex, so even the create-to-flock gap
        # cannot cause an in-progress publisher to be mistaken for a crash.
        entries = self._entries()
        serial = entries[-1][0] + 1 if entries else 1
        for name in os.listdir(self.directory):
            if name.startswith('.pending-'):
                try:
                    self._remove_dead(name)
                except PermissionError:
                    pass  # A different UID may have died before chmod(0444).
        fd, pending = tempfile.mkstemp(prefix='.pending-', dir=LOCK_DIR / QUEUE_NAME)
        self.fd = fd
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fchmod(fd, 0o444)  # Readers of other UIDs need only O_RDONLY.
            os.write(fd, (json.dumps(metadata, ensure_ascii=False) + '\n').encode())
            name = f'{serial:020d}-{Path(pending).name[9:]}.ticket'
            # Unique name and serialized publication; rename publishes the
            # already-locked inode atomically, with no extra hard-link window.
            os.rename(Path(pending).name, name,
                      src_dir_fd=self.directory, dst_dir_fd=self.directory)
            self.name = name
        finally:
            try:
                os.unlink(pending)
            except FileNotFoundError:
                pass

    def try_acquire(self, device_fd, metadata):
        if not try_lock(self.directory):
            return False
        published = False
        try:
            if self.fd is None:
                predecessor = self._same_owner_live_ticket(metadata)
                if predecessor is not None:
                    if predecessor != self.owner_predecessor:
                        emit(f'owner {metadata["owner"]} already has live ticket '
                             f'{predecessor}; waiting without another ticket')
                        self.owner_predecessor = predecessor
                    return False
                self._publish(metadata)
                metadata['queue_ticket'] = self.name
                published = True
            for _, name in self._entries():
                if name == self.name:
                    return try_lock(device_fd)
                if not self._remove_dead(name):
                    return False
            raise OSError('own queue ticket disappeared')
        finally:
            fcntl.flock(self.directory, fcntl.LOCK_UN)
            if published:
                emit(f'queued {self.name} for {metadata["owner"]}')

    def close(self):
        try:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            if self.name is not None:
                try:
                    os.unlink(self.name, dir_fd=self.directory)
                except FileNotFoundError:
                    pass
        finally:
            os.close(self.directory)


def read_metadata(fd):
    try:
        return json.loads(os.pread(fd, 65536, 0))
    except (ValueError, OSError):
        return None


def write_metadata(fd, data):
    data['updated_at'] = now()
    encoded = (json.dumps(data, ensure_ascii=False) + '\n').encode()
    os.pwrite(fd, encoded, 0)
    os.ftruncate(fd, len(encoded))


def emit(message):
    print(f'[npu-guard] {message}', file=sys.stderr, flush=True)


def record_failure(metadata, kind, sample):
    """Retain the triggering sample without changing any admission decision."""
    metadata['failure_snapshot'] = {'kind': kind, 'sample': sample}
    emit('failure_snapshot ' + json.dumps(metadata['failure_snapshot'], sort_keys=True))


def group_alive(pgid):
    # Zombies cannot use the NPU and must not prevent releasing a reservation.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def process_table():
    table = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(entry.name)] = {'pid': int(entry.name), 'state': fields[0],
                                      'ppid': int(fields[1]), 'pgid': int(fields[2]),
                                      'start_ticks': int(fields[19])}
        except (OSError, ValueError, IndexError):
            continue
    return table


class OwnedProcesses:
    """Track the workload tree, including setsid and adopted double-fork children.

    Linux subreaping follows ancestry without granting privilege to workloads.
    Submission to a pre-existing external daemon still requires a separate
    controlled scheduler; this is not a device-access enforcement boundary.
    """
    def __init__(self):
        self.supervisor = os.getpid()
        self.preexisting = {(p, r['start_ticks']) for p, r in process_table().items()
                            if r['ppid'] == self.supervisor}
        self.known = {}
        self.history = {}
        self.child_pid = None
        self.previous_subreaper = C.c_int()
        self.libc = C.CDLL(None, use_errno=True)
        self.libc.prctl.argtypes = [C.c_int, C.c_ulong, C.c_ulong, C.c_ulong, C.c_ulong]
        self.libc.prctl.restype = C.c_int
        if self.libc.prctl(37, C.addressof(self.previous_subreaper), 0, 0, 0) != 0:
            raise OSError(C.get_errno(), 'cannot query child subreaper state')
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(C.get_errno(), 'cannot enable child subreaper tracking')

    def attach(self, pid):
        self.child_pid = pid

    def refresh(self):
        table = process_table()
        owned = {pid for pid, start in self.known.items()
                 if pid in table and table[pid]['start_ticks'] == start}
        for pid, row in table.items():
            if row['ppid'] == self.supervisor and (pid, row['start_ticks']) not in self.preexisting:
                owned.add(pid)
        changed = True
        while changed:
            children = {pid for pid, row in table.items() if row['ppid'] in owned}
            changed = not children <= owned
            owned |= children
        self.known = {pid: table[pid]['start_ticks'] for pid in owned}
        self.history.update(self.known)
        self.alive = {pid: table[pid] for pid in owned if table[pid]['state'] != 'Z'}
        for pid in owned:
            if pid != self.child_pid and table[pid]['state'] == 'Z' and table[pid]['ppid'] == self.supervisor:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        return bool(self.alive)

    def owns(self, item):
        start = self.history.get(item['pid'])
        return start is not None and item.get('start_ticks') == start

    def send(self, sig):
        self.refresh()
        for pid, row in self.alive.items():
            try:
                # A pidfd pins the process while its identity is checked, so a
                # recycled PID never receives a signal intended for a child.
                handle = os.pidfd_open(pid)
                try:
                    if proc_identity(pid)['start_ticks'] == row['start_ticks']:
                        signal.pidfd_send_signal(handle, sig)
                finally:
                    os.close(handle)
            except ProcessLookupError:
                pass

    def close(self):
        self.libc.prctl(36, self.previous_subreaper.value, 0, 0, 0)


def foreign_processes(sample, pgid, owned=None):
    """Only this invocation's tracked descendants/group own its reservation."""
    foreign = set()
    # Runtime monitoring uses registered contexts. Read-only status probes can
    # briefly open manager FDs and must not abort a running workload.
    for item in sample.get('pids', []):
        pid = item['pid']
        if owned is not None and owned.owns(item):
            continue
        try:
            group = item.get('pgid')
            if group is None:
                group = os.getpgid(pid)
            if group != pgid:
                foreign.add(pid)
        except ProcessLookupError:
            continue
    return sorted(foreign)


def execute(fd, metadata, command, timeout, grace=5, sample_fn=probe,
            on_conflict='record'):
    """Keep the reservation through foreground work and descendants.

    The child inherits the same flock FD: killing only this supervisor does
    not release the reservation while the immediate child is still alive.
    Surviving runtime processes are also caught by the next physical probe.
    """
    received = []
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[sig] = signal.signal(sig, lambda number, frame: received.append(number))
    child = None
    owned = None
    try:
        if on_conflict not in ('record', 'terminate'):
            raise ValueError('on_conflict must be record or terminate')
        owned = OwnedProcesses()
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='0', ASCEND_VISIBLE_DEVICES='0')
        # A fresh session permits terminating only this invocation's process group.
        child = subprocess.Popen(command, pass_fds=(fd,), start_new_session=True, env=env)
        owned.attach(child.pid)
        metadata.update(phase='running', child=proc_identity(child.pid),
                        conflict_policy=on_conflict, conflict_detected=False,
                        visibility_lost=False, performance_valid=True, tracking='linux-subreaper-pidfd')
        write_metadata(fd, metadata)
        started, stopping, stop_code = time.monotonic(), None, None
        next_probe = started + 2
        while True:
            code = child.poll()
            alive = owned.refresh()
            finished = code is not None and not alive
            timed_out = not finished and timeout is not None and time.monotonic() - started >= timeout
            conflict_code = None
            if stopping is None and not received and not timed_out and (finished or time.monotonic() >= next_probe):
                sample = sample_fn()
                if finished:
                    metadata['postflight_checked_at'] = now()
                owned.refresh()
                foreign = foreign_processes(sample, child.pid, owned)
                if foreign:
                    first = not metadata['conflict_detected']
                    metadata.update(conflict_detected=True, performance_valid=False)
                    metadata['external_pids'] = sorted(set(metadata.get('external_pids', [])) | set(foreign))
                    if first:
                        action = 'stopping this workload' if on_conflict == 'terminate' else 'recording conflict; workload continues; performance invalid'
                        emit(f'external Ascend users appeared: {foreign}; {action}')
                        record_failure(metadata, 'external_processes', sample)
                    metadata['last_conflict_at'] = now()
                    write_metadata(fd, metadata)
                    if on_conflict == 'terminate':
                        conflict_code = CONFLICT
                if (sample['state'] == 'unknown' or sample.get('errors') or sample.get('unreadable_proc_pids')
                      or sample.get('pid_query_rc', 0) != 0
                      or sample.get('health', {'rc': 0, 'value': 0}) != {'rc': 0, 'value': 0}):
                    health = sample.get('health', {})
                    unhealthy = health.get('rc') == 0 and health.get('value', 0) != 0
                    terminate = on_conflict == 'terminate' or unhealthy
                    if not metadata['visibility_lost'] or unhealthy:
                        emit('device observation incomplete or unhealthy; ' +
                             ('stopping this workload' if terminate else 'workload continues; performance invalid'))
                        record_failure(metadata, 'device_unhealthy' if unhealthy else 'device_visibility_lost', sample)
                    metadata['visibility_lost'] = True
                    metadata['performance_valid'] = False
                    write_metadata(fd, metadata)
                    if terminate:
                        conflict_code = UNAVAILABLE
                next_probe = time.monotonic() + 2
            if finished:
                workload_code = code if code >= 0 else 128 - code
                metadata['workload_exit_code'] = workload_code
                metadata['workload_completed'] = stop_code is None
                if workload_code:
                    metadata['performance_valid'] = False
                if stop_code is not None:
                    return stop_code
                if received:
                    metadata['performance_valid'] = False
                    return 128 + received[0]
                # Preserve workload failures and prevent contaminated timing
                # from passing as success, including interference at exit.
                return (workload_code or conflict_code
                        or (UNAVAILABLE if metadata['visibility_lost'] else 0)
                        or (CONFLICT if metadata['conflict_detected'] else 0))
            if stopping is None and (received or timed_out or conflict_code):
                sig = received[0] if received else signal.SIGTERM
                stop_code = 128 + sig if received else (conflict_code or 124)
                stopping = time.monotonic()
                metadata.update(phase='stopping', performance_valid=False)
                write_metadata(fd, metadata)
                owned.send(sig)
            elif stopping is not None:
                owned.send(signal.SIGKILL if time.monotonic() - stopping >= grace else sig)
            time.sleep(0.1)
    except BaseException:
        # Unexpected observer/logging failures must not leave an unmonitored
        # command behind. Its session/group was created by this invocation.
        if child is not None and owned is not None:
            metadata.update(performance_valid=False, workload_completed=False)
            try:
                owned.send(signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            cleanup_deadline = time.monotonic() + 5
            while owned.refresh() and time.monotonic() < cleanup_deadline:
                owned.send(signal.SIGKILL)
                time.sleep(.05)
        raise
    finally:
        if owned is not None:
            owned.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def run(args, sample_fn=probe):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('provide a foreground command after --')
    profile = device_profile()
    # A private receipt survives the next owner overwriting shared metadata.
    # Reserve its name before starting any workload; never overwrite evidence.
    receipt = None
    if getattr(args, 'receipt', None):
        receipt = open(args.receipt, 'x', encoding='utf-8')
    try:
        fd = open_lock()
    except BaseException:
        if receipt is not None:
            receipt.close()
        raise
    acquired = False
    ticket = None
    metadata = {**proc_identity(os.getpid()), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'owner': args.owner, 'device': 0, 'coordination_scope': profile['scope'],
                'device_node': profile['device_node'], 'command': Path(command[0]).name, 'phase': 'waiting_lock',
                'requested_at': now()}
    deadline = time.monotonic() + args.wait
    last_message, last_report, quiet_since, visibility_since = None, 0, None, None
    try:
        ticket = FifoTicket()
        first_attempt = True
        while True:
            if not first_attempt and time.monotonic() >= deadline:
                emit('wait deadline reached; no workload launched')
                return BUSY
            first_attempt = False
            if not acquired:
                acquired = ticket.try_acquire(fd, metadata)
                if acquired:
                    metadata.update(phase='waiting_device', acquired_at=now())
                    write_metadata(fd, metadata)
            if acquired:
                sample = sample_fn()
                state = sample['state']
                message = f"device {state}: {'; '.join(sample['reasons'])}"
                if state == 'unknown':
                    # Short exec/credential transitions should not reject an
                    # entire queued job. Retry full observations for at most
                    # five seconds; never launch with an incomplete snapshot.
                    retryable = (sample.get('unreadable_proc_pids') and not sample.get('errors')
                                 and sample.get('health') == {'rc': 0, 'value': 0}
                                 and sample.get('pid_query_rc') == 0)
                    if visibility_since is None:
                        visibility_since = time.monotonic()
                    if retryable and time.monotonic() < min(deadline, visibility_since + 5):
                        quiet_since = None
                        if message != last_message:
                            emit(message + '; retrying complete observation')
                            last_message = message
                        time.sleep(min(args.poll, max(0, deadline - time.monotonic())))
                        continue
                    emit(message)
                    record_failure(metadata, 'device_unavailable_before_launch', sample)
                    write_metadata(fd, metadata)
                    return UNAVAILABLE
                visibility_since = None
                if state == 'idle':
                    if quiet_since is None:
                        quiet_since = time.monotonic()
                    if time.monotonic() - quiet_since >= args.idle_seconds:
                        emit(f"acquired device 0 for {args.owner}")
                        result = execute(fd, metadata, command, args.timeout, sample_fn=sample_fn,
                                         on_conflict=getattr(args, 'on_conflict', 'record'))
                        metadata.update(phase='finished', exit_code=result)
                        return result
                else:
                    quiet_since = None
                    message += f"; pids={[p['pid'] for p in sample.get('pids', [])]}"
            else:
                holder = read_metadata(fd)
                message = f'waiting for reservation: {holder}'
            current = time.monotonic()
            if message != last_message or current - last_report >= 30:
                emit(message)
                last_message, last_report = message, current
            if current >= deadline:
                emit('wait deadline reached; no workload launched')
                return BUSY
            time.sleep(min(args.poll, max(0, deadline - current)))
    finally:
        try:
            if acquired:
                if metadata['phase'] != 'finished':
                    metadata['phase'] = 'aborted' if 'child' in metadata else 'released_without_launch'
                write_metadata(fd, metadata)
        finally:
            # Never LOCK_UN: a surviving child may share this open file description.
            try:
                try:
                    if receipt is not None:
                        json.dump(metadata, receipt, ensure_ascii=False, indent=2)
                        receipt.write('\n')
                        receipt.flush()
                        os.fsync(receipt.fileno())
                finally:
                    if receipt is not None:
                        receipt.close()
                    os.close(fd)
            finally:
                if ticket is not None:
                    ticket.close()


def status(as_json):
    sample = probe()
    try:
        fd = open_lock()
    except OSError as exc:
        sample['reservation'] = {'error': str(exc)}
        sample['state'] = 'unknown'
        sample['reasons'].append('shared lock is unavailable')
    else:
        try:
            free = try_lock(fd)
            sample['reservation'] = {'locked': not free, 'metadata': read_metadata(fd)}
            if not free:
                sample['state'] = 'busy'
                sample['reasons'].append('cooperative reservation is held')
        finally:
            os.close(fd)
    if as_json:
        print(json.dumps(sample, ensure_ascii=False))
    else:
        print(f"device 0: {sample['state']}")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    return {'idle': 0, 'busy': BUSY, 'unknown': UNAVAILABLE}[sample['state']]


def finite_number(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be a finite nonnegative number')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    subs.add_parser('init', help='root: initialize the host lock after boot')
    state = subs.add_parser('status', help='read-only device and reservation snapshot')
    state.add_argument('--json', action='store_true')
    job = subs.add_parser('run', help='reserve, wait for physical availability, run foreground command')
    job.add_argument('--owner', required=True, help='agent/task name (no secrets)')
    job.add_argument('--wait', type=finite_number, default=600, help='acquisition deadline in seconds (default 600)')
    job.add_argument('--poll', type=finite_number, default=1)
    job.add_argument('--idle-seconds', type=finite_number, default=3, help='continuous quiet window (default 3)')
    job.add_argument('--timeout', type=finite_number, help='workload wall time; stops only its tracked descendants')
    job.add_argument('--on-conflict', choices=('record', 'terminate'), default='record',
                     help='record (default): finish workload, mark performance invalid, return76; terminate: stop owned task')
    job.add_argument('--receipt', help='new private JSON receipt, written before releasing the reservation')
    job.add_argument('command', nargs=argparse.REMAINDER)
    subs.add_parser('_probe', help=argparse.SUPPRESS)
    subs.add_parser('serve-probes', help='root: serve fixed read-only device queries over the protected Unix socket')
    args = parser.parse_args()
    try:
        if args.action == '_probe':
            print(json.dumps(raw_probe(), ensure_ascii=False), flush=True)
            os._exit(0)
        if args.action == 'serve-probes':
            serve_probes()
            return 0
        if args.action == 'init':
            fd = open_lock()
            os.close(fd)
            os.close(open_queue())
            print(LOCK_DIR / LOCK_NAME)
            return 0
        if args.action == 'status':
            return status(args.json)
        if args.poll <= 0:
            parser.error('--poll must be positive')
        return run(args)
    except (OSError, ValueError) as exc:
        emit(str(exc))
        return UNAVAILABLE
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
