"""Explicit opt-in only. No automatic rerun after a capture/decoder failure."""
from pathlib import Path
import hashlib,json

def choose(*, opt_in=False, stateless=False, single_stream=False, metric='device', reset=None):
    reasons=[]
    if not opt_in: reasons.append('not explicitly opted in')
    if not stateless or reset is not None: reasons.append('stateful/reset not validated')
    if not single_stream: reasons.append('single-stream contract not established')
    if metric!='device': reasons.append('host-completed equivalence not established')
    return {'route':'original' if reasons else 'restricted_session','reasons':reasons,
            'metric':metric,'automatic_retry':False,'raw_audit_required':True}

def make_selected_timer(original_timer,evidence_dir,*,opt_in=False,stateless=False,single_stream=False,metric='device'):
    def timer(torch,torch_npu,fn,inputs,warmup,repeats,reset=None):
        decision=choose(opt_in=opt_in,stateless=stateless,single_stream=single_stream,metric=metric,reset=reset)
        root=Path(evidence_dir);root.mkdir(parents=True,exist_ok=False)
        decision['module_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
        (root/'selection.json').write_text(json.dumps(decision,indent=2))
        if decision['route']=='original':
            # Selection happens before any call; original timer controls supported reset semantics.
            result=original_timer(torch,torch_npu,fn,inputs,warmup,repeats,reset)
        else:
            from .capture import make_timer
            # Exceptions intentionally propagate; no second execution after partial measurement.
            result=make_timer(root/'session')(torch,torch_npu,fn,inputs,warmup,repeats,reset)
        result['protocol_selection']=decision
        return result
    return timer
