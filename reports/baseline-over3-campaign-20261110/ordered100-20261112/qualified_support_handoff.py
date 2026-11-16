"""Confirm a successful support build was consumed by full formal acceptance."""
from pathlib import Path
import json,hashlib

def consumed_support(directory,state,entry):
 try:
  d=Path(directory);read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();registry=read(d/'QUALIFIED_SUPPORT_HANDOFFS.json')
  for cert in registry['rows']:
   if cert['entry']!=entry or cert['pid']!=state.get('pid') or cert['tag']!=state.get('tag'):continue
   f=d/cert['build_state'];assert sha(f)==cert['build_state_sha256'] and read(f)==state;assert state['exit_code']==0 and state['guard']['performance_valid'] and state['guard']['workload_exit_code']==0
   overlay=d/cert['qualified_overlay'];assert sha(overlay)==cert['qualified_overlay_sha256'];rows=read(overlay)['rows'];accepted={r['ordinal']:r for r in rows if r.get('full_checker_15raw_guard_valid')};assert set(cert['qualified_scope'])<=set(accepted)
   for i in cert['qualified_scope']:
    row=accepted[i];cp=d/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256'];event=read(cp);assert event['guard']['performance_valid'] and event['guard']['workload_exit_code']==0 and event['guard']['phase']=='finished';r=next(x for x in event['batch']['results'] if x['ordinal']==i);assert r['status']=='raw_checked_pending_guard' and len(r['timing']['latency_samples_ms'])==15
   assert sha(d/cert['source_device_attribution'])==cert['source_device_attribution_sha256'];return cert
 except (OSError,KeyError,ValueError,TypeError,AssertionError):return None
 return None
