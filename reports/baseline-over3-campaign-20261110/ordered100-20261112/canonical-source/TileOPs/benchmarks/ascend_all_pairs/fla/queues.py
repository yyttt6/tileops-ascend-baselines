"""CPU-only queues for every exact case with an implemented eligible route."""
import json
from pathlib import Path
from environment import mode_for

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla'


def main():
    inventory=json.loads((EVIDENCE.parent/'inventory.json').read_text())
    namespace={'__file__':str(Path(__file__).with_name('__init__.py'))}
    exec(compile(Path(__file__).with_name('__init__.py').read_text(),__file__,'exec'),namespace)
    providers=namespace['PROVIDERS']
    queues={}
    excluded=[]
    for number,case in enumerate(inventory['cases']):
        op=case['op']
        if op not in providers:continue
        p=case['params'];t=case['tensors'];mode='triton'
        reason=None
        if op in ('GLAFwdOp','GLARecurrentFwdOp','GLAChunkFwdOp','DeltaNetChunkFwdOp','DeltaNetChunkBwdOp') and t['q'][1]!='bfloat16':
            reason='Inspected original native API requires BF16; FP16-to-BF16 is lossy and prohibited.'
        if op=='VectorNormFwdOp' and p.get('ord',2) not in (1,2):
            reason='No complete infinity-norm adapter registered; this factory supports only original L1/L2 compositions.'
        if reason:
            excluded.append(dict(op=op,case_id=case['case_id'],reason=reason,device_attempt_required=False))
            continue
        if op in ('Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp'):
            taps=1
            for size in t['weight'][0][1:]:taps*=size
        if op=='BmmFwdOp':
            mode='triton' if t['a'][0][-1]>128 else 'gdn'
        if op in ('GemmINT8W8A8FwdOp','GemmFP8FwdOp','BmmFP8FwdOp','GLAFwdOp','GLARecurrentFwdOp','ExpFwdOp','Expm1FwdOp'):mode='gdn'
        if op in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp'):mode='causal'
        if op in ('DeltaNetRecurrentFwdOp','SigmoidFwdOp','TanhFwdOp'):mode='recurrent'
        if op=='DeltaNetFwdOp':mode='recurrent' if t['q'][1]=='bfloat16' and t['q'][0][1]<=8 else 'gdn'
        if op=='GDNFwdOp':
            mode='recurrent' if t['q'][1]=='bfloat16' and (p.get('use_gate_in_kernel') or t['q'][0][1]<=8) else 'gdn'
        if op=='KDAFwdOp':mode='recurrent' if t['q'][0][1]<=8 else 'chunk_kda'
        mode=mode_for(op,t,p)
        queues.setdefault(mode,[]).append(dict(op=op,case_id=case['case_id'],mode=mode,receipt=f'full-{number:04d}.json'))
    for mode,rows in queues.items():
        (EVIDENCE/f'queue-{mode}.json').write_text(json.dumps(rows,indent=2)+'\n')
    (EVIDENCE/'queue-static-exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    print({mode:len(rows) for mode,rows in queues.items()},'static_exclusions',len(excluded))


if __name__=='__main__':main()
