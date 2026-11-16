"""CPU-only queue generation from exact manifest inventory, no synthetic cases."""
import json,math,sys
from screen import screen
from pathlib import Path
ROOT=Path('/root/dyq/workspace');E=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
x=json.loads((E.parent/'inventory.json').read_text());coverage=json.loads((E/'coverage.json').read_text())
revision=('_'+sys.argv[1]) if len(sys.argv)>1 else ''
ops={o['op'] for o in coverage['operators'] if o['adapter']}
def cost(c):
    tensors=c.get('tensors',{}) or {k:v for k,v in c['params'].items() if isinstance(v,list) and len(v)==2 and isinstance(v[-1],str)}
    return sum(math.prod(t[0]) for t in tensors.values())
historical={(o['op'],c['case_id']) for o in coverage['operators'] for c in o['cases'] if c['status']=='verified_historical'}
def capable(c):
    if (c['op'],c['case_id']) in historical:return False
    if screen(c):return False
    p=c['params'];t=c.get('tensors',p);op=c['op']
    if op in ('GQAPagedFwdOp','GQAVarlenFwdOp'):
        return t['q'][1] in ('float16','bfloat16') and p.get('softcap') is None and p.get('pos_encoding_mode','none')=='none' and not any(k in t for k in ('sinks','q_scale','k_scale','v_scale','rope_cos','rope_sin'))
    if op in ('KDAFwdOp','GDNFwdOp','DeltaNetFwdOp','DeltaNetChunkFwdOp','DeltaNetRecurrentFwdOp') and t['q'][1]!='bfloat16':return False
    if op=='GeluTanhAndMulFwdOp':return t['x'][1]=='float16'
    if op=='LayerNormFwdOp':return math.prod(p['normalized_shape'])<=1024
    if op in ('KDAFwdOp','GDNFwdOp','DeltaNetFwdOp') and p.get('use_qk_l2norm_in_kernel'):return False
    if op in ('GroupNormFwdOp','InstanceNormFwdOp'):
        shape=t['x'][0];groups=p.get('num_groups',shape[1]);return math.prod(shape[1:])//groups<=1024 and math.prod(shape[1:])//groups%2==0 and p.get('use_input_stats',True)
    return True
out=[]
for op in sorted(ops,key=lambda o:(o!='BmmFwdOp',o)):
    cases=sorted([c for c in x['cases'] if c['op']==op and capable(c)],key=cost)
    if not cases:continue
    c=cases[0];out.append(['--op',op,'--case-id',c['case_id'],'--library','ops-transformer','--output',str(E/('probe'+revision)/f'{op}__{c["case_id"]}.json')])
# One additional route probe: the softmax branch uses a different native API.
softmax_cases=[c for c in x['cases'] if c['op']=='FusedTopKFwdOp' and c['params']['scoring_func']=='softmax' and capable(c)]
if softmax_cases:
    c=min(softmax_cases,key=cost)
    out.append(['--op',c['op'],'--case-id',c['case_id'],'--library','ops-transformer','--output',str(E/('probe'+revision)/f'{c["op"]}__{c["case_id"]}.json')])
lookup={(c['op'],c['case_id']):cost(c) for c in x['cases']}
out.sort(key=lambda a:(a[1]!='BmmFwdOp',lookup[a[1],a[3]]))
(E/'probe_queue.json').write_text(json.dumps(out,indent=2)+'\n')
print([(a[1],a[3]) for a in out])

# Every eligible original case is retained for resumption after representative probes.
full=[]
for c in sorted([c for c in x['cases'] if c['op'] in ops and capable(c)],key=lambda c:(c['op'],cost(c))):
    full.append(['--op',c['op'],'--case-id',c['case_id'],'--library','ops-transformer','--output',str(E/('manifest'+revision)/f'{c["op"]}__{c["case_id"]}.json')])
(E/'full_queue.json').write_text(json.dumps(full,indent=2)+'\n')
print('Full original-manifest cases:',len(full))
