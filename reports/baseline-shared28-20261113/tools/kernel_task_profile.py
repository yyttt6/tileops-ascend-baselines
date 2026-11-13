"""Sum distinct kernel task spans; count AIC/AIV roles of one kernel once."""
import collections
import hashlib
import json
import math
from pathlib import Path
import struct


PARSER_REVISION = "v3-task-coverage-20261103"

def trace_digest(directory):
    root = Path(directory)
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and not p.name.endswith(('.done', '.complete')) and any(
                token in p.name for token in ('ffts_profile', 'hash_dic.slice_', 'task_track.slice_', 'stars_soc.data.', 'info.json')):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def parse_profile(prof_path, repeats, *, seq_to_name=None, exclude_name_marker=None,
                  exclude_names=frozenset(), require_marker_seen=False):
    root = Path(prof_path)
    if repeats <= 0:
        raise ValueError('Positive repeat count required')
    # Strictly support the audited single-device platform-15 trace format.
    infos = list(root.rglob('info.json.0'))
    if len(infos) != 1:
        raise ValueError('Exactly one device info record required')
    info = json.loads(infos[0].read_text())
    if str(info.get('platform_version')) != '15':
        raise ValueError('Unsupported STAR platform')
    mhz = float(info['DeviceInfo'][0]['hwts_frequency'])
    if not math.isfinite(mhz) or mhz <= 0:
        raise ValueError('Invalid hardware timestamp frequency')
    hashes = {}
    for p in root.rglob('*hash_dic.slice_*'):
        if p.name.endswith(('.done', '.complete')):
            continue
        for line in p.read_text().splitlines():
            a, sep, b = line.partition(':')
            if sep:
                try:
                    key = int(a)
                except ValueError:
                    continue
                hashes[key % (1 << 64)] = b
    names = dict(seq_to_name or {})
    named = set()
    streams = collections.defaultdict(set)
    for p in root.rglob('*task_track.slice_*'):
        if p.name.endswith(('.done', '.complete')):
            continue
        raw = p.read_bytes()
        if len(raw) % 64:
            raise ValueError('Truncated task track: ' + str(p))
        for offset in range(0, len(raw), 64):
            v = struct.unpack_from('<8q', raw, offset)
            seq = (v[3] >> 32) & 65535
            streams[seq].add((v[3] >> 16) & 65535)
            if v[5]:
                named.add(seq)
                name = hashes.get(v[5] % (1 << 64), '')
                if name:
                    if seq in names and names[seq] != name:
                        raise ValueError('Ambiguous task name')
                    names[seq] = name
    if not named:
        raise ValueError('Missing named task coverage records')
    if len({stream for seq in named for stream in streams[seq]}) > 1:
        raise ValueError('Multi-stream traces require a separately validated overlap decoder')
    groups = collections.defaultdict(list)
    aic_cycles = aiv_cycles = raw_count = 0
    for p in root.rglob('ffts_profile*'):
        if p.name.endswith(('.done', '.complete')):
            continue
        raw = p.read_bytes()
        if len(raw) % 128:
            raise ValueError('Truncated FFTS record: ' + str(p))
        for offset in range(0, len(raw), 128):
            v = struct.unpack_from('<16q', raw, offset)
            if v[1] == 0:
                continue
            # CANN Utils.get_func_type defines only the low six header bits
            # as the record function. Higher packet flags are not core roles.
            kind = v[0] & 63
            if kind != 0x2A:
                raise ValueError('Unsupported FFTS record function: ' + hex(kind))
            if v[15] < v[14] or v[14] <= 0:
                raise ValueError('Invalid kernel timestamp')
            role = 'AIV' if v[2] >= 1 << 31 else 'AIC'
            raw_count += 1
            if role == 'AIV':
                aiv_cycles += v[1]
            else:
                aic_cycles += v[1]
            # Task identity is in the upper header word. Packet flags in the
            # lower word may differ between records of the same task. Retain
            # the FFTS context and reject ambiguous stream mappings below.
            key = ((v[0] >> 32) & 0xFFFFFFFF, (v[2] >> 16) & 65535)
            groups[key].append((role, v[14], v[15]))

    tasks = []
    excluded_count = excluded_ns = 0
    marker_seen = False
    for (task_id, context), records in groups.items():
        seq = task_id & 65535
        roles = {r[0] for r in records}
        if len(streams[seq]) > 1:
            raise ValueError('Ambiguous multi-stream task ID; retain trace for full decoding')
        for role in roles:
            intervals = {(start, end) for r, start, end in records if r == role}
            if len(intervals) > 1:
                raise ValueError('Repeated task/context identity; do not merge unrelated launches')
        start = min(r[1] for r in records)
        end = max(r[2] for r in records)
        name = names.get(seq, '')
        is_marker = bool(exclude_name_marker) and exclude_name_marker in name
        marker_seen |= is_marker
        if is_marker or name in exclude_names:
            excluded_count += 1
            excluded_ns += end - start
            continue
        tasks.append(dict(task_id=task_id, context_id=context, stream_ids=sorted(streams[seq]),
                          name=name, core_roles=sorted(roles), start_ns=start, end_ns=end,
                          duration_us=(end-start)/1000))
    # STAR v6 start/end packets cover AICPU tasks absent from FFTS.
    # CANN AcsqTaskV6 uses (task_high << 16) | task_low as its unique ID.
    # Reject reuse/ambiguity rather than silently merging unrelated launches.
    ffts_seqs = {task_id & 65535 for task_id, context in groups}
    pending, completed = {}, {}
    star_files = sorted(root.rglob('stars_soc.data.*.slice_*'))
    for p in star_files:
        if p.name.endswith(('.done', '.complete')):
            continue
        raw = p.read_bytes()
        if len(raw) % 32:
            raise ValueError('Truncated STAR record: ' + str(p))
        for v in struct.iter_unpack('<4HQHH3L', raw):
            fn, kind = v[0] & 63, v[0] >> 10
            if fn not in (0, 1):
                continue
            uid = (v[3] << 16) | v[2]
            key = (uid, kind)
            if fn == 0:
                if key in pending or key in completed:
                    raise ValueError('Repeated STAR task identity')
                pending[key] = v[4]
            else:
                if key not in pending:
                    raise ValueError('Unmatched STAR end')
                start = pending.pop(key)
                if v[4] < start:
                    raise ValueError('Invalid STAR timestamp')
                completed[key] = (start, v[4])
    if pending:
        raise ValueError('Unclosed STAR tasks')
    missing = named - ffts_seqs
    covered = set()
    added_count = 0
    for (uid, kind), (start, end) in completed.items():
        if uid not in missing:
            if uid > 65535 and (uid & 65535) in missing:
                raise ValueError('Ambiguous truncated task identity')
            continue
        if kind not in (4, 5):
            raise ValueError('Unsupported named non-FFTS task type: ' + str(kind))
        if uid in covered:
            raise ValueError('Multiple STAR records for missing task')
        covered.add(uid)
        name = names.get(uid, '')
        is_marker = bool(exclude_name_marker) and exclude_name_marker in name
        marker_seen |= is_marker
        ns = (end - start) * 1000 / mhz
        if is_marker or name in exclude_names:
            excluded_count += 1
            excluded_ns += ns
            continue
        added_count += 1
        tasks.append(dict(task_id=uid, context_id=None, stream_ids=sorted(streams[uid]),
                          name=name, core_roles=['AICPU'], source='STAR', task_type=kind,
                          start_ns=start*1000/mhz, end_ns=end*1000/mhz,
                          duration_us=(end-start)/mhz))
    if missing - covered:
        raise ValueError('Uncovered named tasks: ' + str(sorted(missing-covered)))
    if not tasks:
        raise ValueError('No measured device activity')
    if require_marker_seen and exclude_name_marker and not marker_seen:
        raise ValueError('Flush marker not resolved')
    # Subtract STAR integer ticks before scaling; absolute floating timestamps
    # lose low bits on long-running hosts. duration_us retains the interval.
    total_ns = sum(t['duration_us'] * 1000 for t in tasks)
    return dict(parser_revision=PARSER_REVISION, profile_named_task_coverage_complete=True,
                profile_aicpu_task_count=added_count/repeats, latency_ms=total_ns/repeats/1e6, profile_kernel_us=total_ns/repeats/1000,
                profile_kernel_count=len(tasks)/repeats, profile_raw_record_count=raw_count/repeats,
                profile_aic_cycles=aic_cycles/repeats, profile_aiv_cycles=aiv_cycles/repeats,
                profile_excluded_kernel_count=excluded_count/repeats,
                profile_excluded_kernel_us=excluded_ns/repeats/1000,
                profile_kernel_tasks=tasks)
