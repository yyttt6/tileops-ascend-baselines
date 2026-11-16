"""CPU-only acceptance: full outputs, preserved inputs, own compile/launch chain."""
import datetime,hashlib,json,re,bisect
from pathlib import Path
ROOT=Path('/root/dyq/workspace')
E=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
_hash_cache={}
def sha(p):
    stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
    if key not in _hash_cache:_hash_cache[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    return _hash_cache[key]
def preserved(path,expected,receipt):
    candidate=Path(receipt.get('source_snapshots',{}).get(source_key(path),str(path)))
    if candidate.is_file() and sha(candidate)==expected:return candidate
    snapshot=ROOT/'.cache/all-pairs/ops_transformer/source_snapshots'/str(expected)/path.name
    if snapshot.is_file() and sha(snapshot)==expected:return snapshot
    return None
def source_key(path):
    p=Path(path)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def instant(text):
    match=re.search(r':(\d{4}-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',text)
    if match:return match[1]+'T'+match[2]+'.'+match[3]+match[4]+'+00:00'
def main():
    compiled={};events={};launches_by_pid={};seen_launches=set()
    for f in E.rglob('*.native.jsonl'):
        receipt_path=f.with_name(f.name.replace('.native.jsonl','.json'))
        try:compile_receipt=json.loads(receipt_path.read_text())
        except (OSError,ValueError):compile_receipt={}
        records=[]
        for line in f.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            txt=r.get('text','');records.append(r)
            if 'StreamLaunchKernel' in txt:
                pid=re.search(r'RUNTIME\((\d+),',txt)
                ts=instant(txt)
                key=(r.get('file'),r.get('line'),txt)
                if pid and ts and key not in seen_launches:
                    seen_launches.add(key)
                    launches_by_pid.setdefault(int(pid[1]),[]).append((ts,r | {'capture_file':str(f)}))
            if 'LogTaskDetails' not in txt or 'result[success]' not in txt:continue
            source=re.search(r'File\[([^]]+)\]',txt);op=re.search(r'opType\[([^]]+)\]',txt);kernel=re.search(r'kernel\[([^]]+)\]',txt)
            if not all((source,op,kernel)):continue
            if not any(v in source[1] for v in ('classicpilot_transformer_impl/dynamic/','allpairs_transformer_impl/dynamic/','causalpairs_transformer_impl/dynamic/','kdapairs_transformer_impl/dynamic/','attentionpairs_transformer_impl/dynamic/','mlapairs_transformer_impl/dynamic/','selectionpairs_transformer_impl/dynamic/')):continue
            compiled.setdefault(kernel[1],[]).append({'op':op[1],'source':source[1],
                'source_sha256':compile_receipt.get('source_sha256',{}).get(source_key(source[1])),
                'event':r,'event_file':str(f)})
        events[str(f)]=records
    launch_times={}
    for pid,records in launches_by_pid.items():
        records.sort(key=lambda item:item[0])
        launch_times[pid]=[item[0] for item in records]
    summary=[]
    for f in E.rglob('*.json'):
        if f.name.endswith('.ownership.json'):continue
        try:r=json.loads(f.read_text())
        except ValueError:continue
        if not isinstance(r,dict) or r.get('status')!='verified' or r.get('library')!='ops-transformer':continue
        hashes=r.get('source_sha256',{});mismatches=[]
        for key,expected in hashes.items():
            p=ROOT/key
            if preserved(p,expected,r) is None:mismatches.append(key)
        required={c['op'] for c in r.get('native_graph_calls',[])};proven=set();launches=[]
        # CANN flushes buffered logs after a case can have completed. Capture
        # files are transport boundaries, not execution boundaries. Keep the
        # exact runtime PID and receipt time interval checks across all files.
        times=launch_times.get(r['runtime_pid'],[])
        start=bisect.bisect_left(times,r['started_at']);end=bisect.bisect_right(times,r['finished_at'])
        for _,event in launches_by_pid.get(r['runtime_pid'],[])[start:end]:
            txt=event['text'];ts=instant(txt)
            if 'StreamLaunchKernel' not in txt or f"({r['runtime_pid']}," not in txt:continue
            if not ts or not r['started_at']<=ts<=r['finished_at']:continue
            name=re.search(r'kernel_name=([^,\s]+)',txt)
            if not name:continue
            matches=[(name[1],compiled[name[1]])] if name[1] in compiled else [(k,v) for k,v in compiled.items() if name[1].startswith(k)]
            for kernel,records in matches:
                for record in records:
                    p=Path(record['source']);key=source_key(p)
                    if record['source_sha256']!=hashes.get(key):continue
                    if preserved(p,hashes.get(key),r) is None:continue
                    proven.add(record['op'])
                    if len(launches)<100:launches.append({'compile':record,'launch':event})
            if required<=proven:break
        dispatch=[];direct_numeric=False
        for call in r.get('native_dispatch_calls',[]):
            source=Path(call.get('source',''));binary=Path(call.get('binary',''))
            ok=call.get('dispatch_owner_checked') and call.get('returned')==0 and source.is_file() and binary.is_file() and hashes.get(source_key(source))==sha(source) and hashes.get(source_key(binary))==sha(binary)
            dispatch.append({'call':call,'source_and_binary_accepted':bool(ok)})
            if ok and not call['op'].endswith('Metadata'):direct_numeric=True
        verification=r.get('verification',{});complete=verification.get('checked_outputs',0)>0 and verification.get('checked_outputs')==verification.get('total_outputs')
        dispatched=all(item['source_and_binary_accepted'] for item in dispatch)
        protection=r.get('input_protection')=='Original workloads.numerics.Request(preserve_inputs=True)'
        synced=r.get('source_attribution',{}).get('runtime_synchronized')
        accepted=bool(complete and protection and synced and not mismatches and required<=proven and dispatched and (required or direct_numeric))
        output={'schema':'ops-transformer-own-native-acceptance/v1','accepted':accepted,'numerical_receipt':str(f),'numerical_receipt_sha256':sha(f),'full_manifest_outputs':complete,'input_preservation':protection,'runtime_synchronized':synced,'source_hash_mismatches':mismatches,'required_graph_apis':sorted(required),'proven_graph_apis':sorted(proven),'missing_graph_apis':sorted(required-proven),'compile_and_launch_evidence':launches,'direct_native_dispatch':dispatch,'audit_script_sha256':sha(Path(__file__))}
        f.with_suffix('.ownership.json').write_text(json.dumps(output,indent=2)+'\n');summary.append([r['op'],r['case_id'],accepted,sorted(required-proven)])
    print(json.dumps(summary))
if __name__=='__main__':main()
