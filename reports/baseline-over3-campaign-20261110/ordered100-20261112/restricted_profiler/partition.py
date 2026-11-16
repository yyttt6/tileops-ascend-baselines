"""Experimental partitioner for existing full32 parser, which ALREADY includes AICPU.
Fail closed for multi-stream, repeated/truncated task IDs and boundary ambiguity.
Never call CI decode().added_aicpu on top of full32 tasks.
"""
import collections,struct
from pathlib import Path
MARKER='_tilelang_profiler_cache_flush'
def partition_tasks(tasks,repeats,star_intervals=None):
 if repeats<=0:raise ValueError('positive repeats required')
 if any(len(t['stream_ids'])!=1 or not t['name'] or t['name'].startswith('<unknown') for t in tasks):raise ValueError('ambiguous task or stream')
 if len({t['stream_ids'][0] for t in tasks})!=1:raise ValueError('multi-stream not validated')
 ids=[t['task_id'] for t in tasks]
 if len(set(ids))!=len(ids) or any(i<0 or i>65535 for i in ids):raise ValueError('repeated/wrapped task identity')
 markers=sorted([t for t in tasks if MARKER in t['name']],key=lambda t:t['start_ns'])
 if len(markers)!=2*repeats:raise ValueError('marker count')
 seq=[t['task_id'] for t in markers]
 if seq!=sorted(seq):raise ValueError('marker sequence/time disagreement')
 samples=[];assigned=set()
 for i in range(repeats):
  left,right=markers[2*i:2*i+2]
  body=[t for t in tasks if left['task_id']<t['task_id']<right['task_id']]
  if not body:raise ValueError('empty window')
  for t in body:
   if t.get('source')=='STAR':
    # Use STAR markers in STAR ticks; never compare converted STAR absolute time to FFTS ns.
    if star_intervals is None:raise ValueError('STAR clock boundaries required')
    li,ri,ti=(star_intervals[x['task_id']] for x in (left,right,t))
    if not li[1]<=ti[0]<=ti[1]<=ri[0]:raise ValueError('STAR body crosses STAR marker boundary')
   elif not left['end_ns']<=t['start_ns']<=t['end_ns']<=right['start_ns']:
    raise ValueError('FFTS body crosses FFTS marker boundary')
  if any(t['task_id'] in assigned for t in body):raise ValueError('reused body')
  assigned.update(t['task_id'] for t in body)
  us=sum(t['duration_us'] for t in body)
  if us<=0:raise ValueError('nonpositive sample')
  samples.append({'latency_ms':us/1000,'profile_kernel_us':us,'profile_kernel_count':len(body),'profile_aicpu_task_count':sum('AICPU' in t['core_roles'] for t in body),'profile_kernel_tasks':body,'profile_excluded_kernel_count':0,'profile_excluded_kernel_us':0,'boundary':{'start':left,'end':right}})
 extra=[t for t in tasks if t['task_id'] not in assigned]
 # Reject any unexpected non-marker activity between windows; no reset in initial stateless trial.
 if any(MARKER not in t['name'] for t in extra):raise ValueError('unassigned non-marker task')
 return samples,{'preparation_and_markers':extra,'assigned_tasks':len(assigned),'total_tasks':len(tasks)}
def partition(path,repeats):
 from tools import kernel_task_profile as parser
 whole=parser.parse_profile(path,1)
 if not whole['profile_named_task_coverage_complete']:raise ValueError('coverage incomplete')
 star=decode_star_intervals(path)
 samples,proof=partition_tasks(whole['profile_kernel_tasks'],repeats,star)
 proof['star_clock_intervals']={str(k):list(v) for k,v in star.items()}
 proof['boundary_clock_policy']='FFTS task bounds against FFTS markers; STAR tasks against STAR markers in integer ticks; no cross-domain absolute comparison'
 cycles=collections.defaultdict(lambda:[0,0])
 for p in Path(path).rglob('ffts_profile*'):
  if p.name.endswith(('.done','.complete')):continue
  for v in struct.iter_unpack('<16q',p.read_bytes()):
   if v[1]:cycles[(v[0]>>32)&0xffffffff][int(v[2]>=1<<31)]+=v[1]
 for s in samples:
  s['profile_aic_cycles']=sum(cycles[t['task_id']][0] for t in s['profile_kernel_tasks'])
  s['profile_aiv_cycles']=sum(cycles[t['task_id']][1] for t in s['profile_kernel_tasks'])
 proof.update(parser_revision=whole['parser_revision'],whole_aicpu_count=whole['profile_aicpu_task_count'],trace_sha256=parser.trace_digest(path))
 return samples,proof

def decode_star_intervals(path):
 pending={};done={}
 for f in sorted(Path(path).rglob('stars_soc.data.*.slice_*')):
  if f.name.endswith(('.done','.complete')):continue
  raw=f.read_bytes()
  if len(raw)%32:raise ValueError('truncated STAR')
  for v in struct.iter_unpack('<4HQHH3L',raw):
   fn,kind=v[0]&63,v[0]>>10
   if fn not in (0,1):continue
   uid=(v[3]<<16)|v[2];key=(uid,kind)
   if fn==0:
    if key in pending or uid in done:raise ValueError('repeated STAR identity')
    pending[key]=v[4]
   else:
    if key not in pending or uid in done:raise ValueError('unmatched or duplicated STAR end')
    start=pending.pop(key)
    if v[4]<start:raise ValueError('invalid STAR interval')
    done[uid]=(start,v[4])
 if pending:raise ValueError('unclosed STAR')
 return done
