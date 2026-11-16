"""Accept new numerical passes only with original vendor compilation and matching launch."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

WS=Path('/root/dyq/workspace')
HERE=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn'

def event_time(event):
    stamp=re.search(r':(\d{4}-\d{2}-\d{2})-(\d{2}:\d{2}:\d{2})\.(\d{3})\.(\d{3})',event.get('text',''))
    return f'{stamp[1]}T{stamp[2]}.{stamp[3]}{stamp[4]}+00:00' if stamp else None

def main():
    events={};compiled={}
    for path in HERE.rglob('*.jsonl'):
        if 'native' not in path.name:continue
        lines=[]
        for line in path.read_text().splitlines():
            try:event=json.loads(line)
            except Exception:continue
            text=event.get('text','');lines.append(event)
            if ('LogTaskDetails' in text and 'result[success]' in text and
                ('allpairs_nn_impl/dynamic/' in text or 'classicpilot_nn_impl/dynamic/' in text)):
                op=re.search(r'opType\[([^]]+)\]',text);kernel=re.search(r'kernel\[([^]]+)\]',text)
                source=re.search(r'File\[([^]]+)\]',text)
                if op and kernel and source:
                    compiled.setdefault(kernel[1],[]).append(dict(op=op[1],source=source[1],event=event,event_file=str(path.relative_to(WS))))
        events[path]=lines
    results=[]
    for path in HERE.rglob('*.json'):
        if path.name.endswith('.audit.json'):continue
        try:row=json.loads(path.read_text())
        except Exception:continue
        if not isinstance(row,dict) or row.get('library')!='ops-nn' or row.get('status')!='verified':continue
        event_file=(path.with_suffix('.native.jsonl') if path.parent==HERE else path.with_name(path.name.split('-')[0]+'-native.jsonl'))
        launches=[];proven=set()
        case_events=[]
        # Continuous capture avoids gaps when CANN flushes a case's launch log after
        # its per-case collector closes. Attribute by the event's own UTC timestamp,
        # never by the later capture time or an adjacent case's launch.
        for event in [*events.get(event_file,[]),*events.get(path.parent/'all-native.jsonl',[])]:
            when=event_time(event)
            if when:
                if row.get('started_at','')<=when<=row.get('finished_at',''):
                    case_events.append(event)
        for event in case_events:
            txt=event['text']
            if 'StreamLaunchKernel' not in txt:continue
            for kernel,records in compiled.items():
                if kernel not in txt:continue
                for record in records:
                    compiled_at=event_time(record['event'])
                    if not compiled_at or compiled_at>event_time(event):continue
                    source=Path(record['source']);key=str(source.relative_to(WS)) if source.is_relative_to(WS) else str(source)
                    expected=row.get('source_sha256',{}).get(key)
                    if expected and source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==expected:
                        proven.add(record['op']);launches.append(dict(compile=record,launch=event))
        api=row.get('api','').split('::')[-1].split(' via ')[0].replace('aclnnForeach','Foreach')
        # APIs are composed explicitly in Candidate.api; descriptive clauses do not add operators.
        expected=set(re.findall(r'\b(?:GatherV2|ScatterUpdate|GatherElements|QuantBatchMatmulV3|MaskedScatter|Foreach\w+|MatMulV3|BatchMatMulV3|Conv2DV2|Conv3DV2|AdaptiveMaxPool2d|AdaptiveMaxPool3d|AdaptiveAvgPool2d|MaxPool3DWithArgmaxV2|AvgPool3D|LayerNormV4|LayerNorm|AddLayerNorm|InplaceAddRmsNorm|RmsNorm|GroupNormV2|BatchNormV3|BatchNormGradV3|LpNormV2|KthValue|UnsortedSegment\w+|SoftmaxV2|LogSoftmaxV2|Relu|Sigmoid|HardSigmoid|HardSwish|LeakyRelu|PRelu|Elu|Selu|Mish|SoftplusV2|GeluV2|Swish|SwiGlu|GeluMul)\b',api))
        verification=row.get('verification',{});complete=verification.get('checked_outputs',0)>0 and verification['checked_outputs']==verification.get('total_outputs')
        changed=[]
        for relative,sha in row.get('source_sha256',{}).items():
            if relative.startswith('TileOPs/benchmarks/ascend_all_pairs/ops_nn/') and relative.endswith('.py'):
                current=WS/relative
                if not current.exists() or hashlib.sha256(current.read_bytes()).hexdigest()!=sha:changed.append(relative)
        accepted=complete and bool(expected) and expected<=proven and not changed
        # Keep at least one concrete compile/launch pair for every required API,
        # before adding other launches; repeated cached records must not crowd it out.
        first_by_api={}
        for item in launches:first_by_api.setdefault(item['compile']['op'],item)
        evidence=[first_by_api[op] for op in sorted(expected) if op in first_by_api]
        evidence.extend(item for item in launches if item not in evidence)
        result=dict(accepted=accepted,adapter_sources_unchanged=not changed,changed_adapter_sources=changed,receipt=str(path.relative_to(WS)),full_manifest_outputs=complete,
                    input_preservation='shared checker Request(preserve_inputs=True)',independent_cpu_reference=row.get('independent_cpu_reference',False),
                    expected_apis=sorted(expected),proven_apis=sorted(proven),missing_apis=sorted(expected-proven),
                    numerical_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    compiler_and_launch_evidence=evidence[:50])
        destination=path.with_suffix('.audit.json')
        payload=json.dumps(result,indent=2)+'\n'
        if destination.exists() and destination.read_text()!=payload:
            shutil.copy2(destination,destination.with_name(destination.name+'.'+str(time.time_ns())+'.prior'))
        destination.write_text(payload);results.append((row['op'],row['case_id'],accepted,sorted(expected-proven)))
    print(json.dumps(results))
if __name__=='__main__':main()
