from pathlib import Path
P=Path('/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112')
s=(P/'summarize_vllm_adaln_fla_defined.py').read_text();start=s.index(' for i in [1117,1118,1119]:');end=s.index("  if r.get('status')",start)
s=s[:start]+''' paths=list(P.glob('rollout-vllm-norm-fla-defined-*.CHECKPOINT.json'))
 seen=set()
 for path in paths:
  label=path.name.removesuffix('.CHECKPOINT.json');event=json.loads(path.read_text())
  for r in event['batch']['results']:
   i=r['ordinal'];assert i in proof['scope'];assert i not in seen;seen.add(i)
'''+''.join(' '+line+'\n' for line in s[end:s.index(' report=',end)].splitlines())+s[s.index(' report=',end):]
s=s.replace('VLLM_ADALN_FLA_DEFINED','VLLM_NORM_FLA_DEFINED').replace('vllm_adaln_fla_defined_api','vllm_norm_fla_defined_api').replace('ordered_vllm_adaln_fla_defined','ordered_vllm_norm_fla_defined').replace('RESOLVED_ADALN_FLA_DEFINED_API','RESOLVED_NORM_FLA_DEFINED_API');(P/'summarize_vllm_norm_fla_defined.py').write_text(s)
# Clone dispatch semantics and source qualification; only finite scope/module names differ.
s=(P/'queue_vllm_adaln_fla_defined.py').read_text().replace('adaln_fla_defined','norm_fla_defined').replace('adaln-fla-defined','norm-fla-defined').replace('ADALN_FLA_DEFINED','NORM_FLA_DEFINED')
s=s.replace("b'queue_sgl_midpoint_api.py' in pathlib.Path('/proc/1526311/cmdline')", "b'queue_vllm_adaln_fla_defined.py' in pathlib.Path('/proc/1550030/cmdline')")
s=s.replace("'indices':[1118,1119,1117],'batches':[[1118],[1119,1117]]", "'indices':[1195,1177,1180,1191,1192,1190,1194,1199],'batches':[[1195],[1177,1180,1191],[1192,1190,1194],[1199]]").replace("'representative':1118", "'representative':1195")
s=s.replace("'SGL_MIDPOINT_API_EXPANSION.json'", "'VLLM_ADALN_FLA_DEFINED_EXPANSION.json'").replace("proof['cases']==36", "proof['cases']==32")
s=s.replace("lineage=json.loads((P/'VLLM_NORM_FLA_DEFINED_PROVENANCE.json').read_text())", "lineage=json.loads((P/'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json').read_text())")
s=s.replace("needle=\"'rollout-sgl-midpoint-api-candidate-'\"", "needle=\"'rollout-vllm-adaln-fla-defined-'\"")
s=s.replace("'source_provenance':'VLLM_NORM_FLA_DEFINED_PROVENANCE.json'", "'source_provenance':'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json'")
# Install separate audited coverage overlay only after predecessor exit, preserving its writer.
needle="  records=[]"
s=s.replace(needle,"  f=P/'effective_coverage.py';old=f.read_text();needle=\"'RESOLVED_ADALN_FLA_DEFINED_API.json'\";assert needle in old\n  if \"'RESOLVED_NORM_FLA_DEFINED_API.json'\" not in old:\n   f.with_name('effective_coverage.before-norm-fla-defined.py').write_text(old);f.write_text(old.replace(needle,needle+\",'RESOLVED_NORM_FLA_DEFINED_API.json'\",1))\n"+needle)
# On numerical failure a bounded worker stops; continue only never-attempted remainder, not the failed case.
start=s.index(" for ids in state['batches']:");end=s.index(" state['status']='all_exact_scope_attempted_preserve_failures'",start)
body=s[start:end];body=body.replace(" for ids in state['batches']:"," pending=list(state['batches']);serial=0\n while pending:\n  ids=pending.pop(0);serial+=1")
body=body.replace("  if state['representative'] in ids:","  attempted={r['ordinal'] for r in event['batch']['results']};assert attempted and attempted.issubset(set(ids))\n  rest=[i for i in ids if i not in attempted]\n  if rest:pending.insert(0,rest)\n  if state['representative'] in ids:")
s=s[:start]+body+s[end:];s=s.replace('candidate-only originalisolated15','candidate-only originalisolated15');(P/'queue_vllm_norm_fla_defined.py').write_text(s)
print('Created separate finite Norm summary/controller; no active-source changes')
