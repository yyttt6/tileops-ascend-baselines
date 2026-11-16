"""CPU-only exact-scope navigation. Classification is not case acceptance."""
from pathlib import Path
import json,collections,hashlib
P=Path(__file__).resolve().parent
q=json.loads((P/'QUEUE.json').read_text());nav=json.loads((P/'LIBRARY_OPERATOR_GROUPS.json').read_text())
methods={
 'natural_feature_layout':dict(issue='Artificial singleton feature dimension changes original API host tiling',action='Preserve natural output shape; invoke unchanged complete API',preconditions=['original host-computed block supported','full broadcast/cast/zeros allocation stays timed','original checker per exactcase'],fallback='original singleton layout',evidence=['VLLM_ARITHMETIC_SELECTION.partial.json','vllm-next-source-review/REVIEW.json','vllm-fused-source-review/REVIEW.json'],state='verified6; Neg3 expansion in progress; fused8 source-reviewed (one nonpower excluded from candidate); other shapes not automatically accepted'),
 'broadcast_and_identity_padding':dict(issue='Redundant padding/copy/layout materialization in broadcast wrappers',action='Skip provable identity padding; select supported complete-API layout',preconditions=['same broadcast semantics','required copies timed','vendor API remains unchanged'],fallback='original wrapper',evidence=['SGL_ADDSUB_SELECTION.json','SGL_CNN_SELECTION.json'],state='exact selection files only; not all arithmetic cases'),
 'complete_api_selection':dict(issue='Per-segment invocation or use of inefficient existing API',action='Compare another existing complete API; include its unused output work and layout conversion',preconditions=['same reduction/segment contract','dtype and precision unchanged','host_roundtrip path measured with original timing mode'],fallback='original API',evidence=['SGL_POOL_SELECTION.json','SGL_MEAN_SELECTION.json','POOL_RAGGED_REVIEW.txt'],state='uniform pool verified; Mean alternate UB failure; ragged remains pending'),
 'equal_width_batching':dict(issue='Repeated complete API calls for equal-width windows',action='Group equal-width windows and call original TopK with original row limit',preconditions=['row cap32768','packing/output order/copies timed','no increased batch after known failure'],fallback='original per-window API',evidence=['DEEPSELECT_ADAPTIVE_SELECTION.json'],state='two exact adaptive cases verified; other reductions not inferred'),
 'native_source_provenance':dict(issue='Captured binary and current rebuild source differ',action='Recover source/build provenance before accepting new shape-plan bridge',preconditions=['no invented platform branch','preserve existing raw evidence'],fallback='retain captured-binary measurements as provenance-pending',evidence=['atvc-natural-server2/BINARY_SOURCE_GAP.json'],state='blocked on provenance, not numerical failure'),
 'deployment_compatibility':dict(issue='Original library JIT/toolchain incompatibility',action='Check original supported build environment without changing kernel/compiler',preconditions=['do not retry same failure conditions','no extrapolated numerical failure'],fallback='deployment pending',evidence=['deepgemm-server1/DEPLOYMENT_ISSUE.json'],state='representative failure only'),
 'source_review_required':dict(issue='Not yet classified from actual implementation',action='Inspect actual factory and source before assigning a correction',preconditions=['name/family does not prove artificial inefficiency'],fallback='unchanged baseline',evidence=[],state='unreviewed')}
def method(r):
 l,o=r['library'],r['op']
 if l=='vllm-ascend' and o in ('AddFwdOp','SubFwdOp','NegFwdOp','FusedAddRMSNormFwdOp'):return 'natural_feature_layout'
 if l=='sgl-kernel-npu' and o in ('AddFwdOp','SubFwdOp','MulFwdOp'):return 'broadcast_and_identity_padding'
 if l=='sgl-kernel-npu' and o in ('MeanFwdOp','MeanPoolingFwdOp'):return 'complete_api_selection'
 if l=='DeepSelect' and o=='AdaptiveMaxPool2dFwdOp':return 'equal_width_batching'
 if l=='atvc':return 'native_source_provenance'
 if l=='DeepGEMM-Ascend':return 'deployment_compatibility'
 return 'source_review_required'
groups=collections.defaultdict(list)
for r in q:groups[(r['stage'],r['library'],method(r),r['op'])].append(r)
source={(r['library'],r['op']):r.get('current_source_literal_references',[]) for r in nav}
out=[]
for (stage,lib,m,op),rs in sorted(groups.items()):
 out.append(dict(stage=stage,library=lib,method=m,op=op,count=len(rs),case_keys=[r['case'] for r in rs],queue_ids=[r['id'] for r in rs],review_status_counts=dict(collections.Counter(r['review_status'] for r in rs)),source_navigation_only=source.get((lib,op),[]),acceptance='Read exact per-case selection evidence; group membership is not acceptance'))
assert sum(x['count'] for x in out)==2208
assert len({(g['library'],g['op'],c) for g in out for c in g['case_keys']})==2208
(P/'REUSE_METHODS.json').write_text(json.dumps(methods,ensure_ascii=False,indent=2)+'\n')
(P/'REUSE_GROUPS.json').write_text(json.dumps({'queue_sha256':hashlib.sha256((P/'QUEUE.json').read_bytes()).hexdigest(),'total_pairs':2208,'groups':out,'policy':'library/API + inefficiency pattern; validate representative then every exactcase; unknown stays unknown'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pairs':2208,'groups':len(out),'method_counts':dict(collections.Counter({m:sum(g['count'] for g in out if g['method']==m) for m in methods}))},indent=2))
