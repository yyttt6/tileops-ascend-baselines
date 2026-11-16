from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pairs=[];pending=[];window_counts={'screening':0,'isolated':0}
for name in ['vllm-silu-server1']:
 D=P/name;rows=json.loads((D/'RESULTS.json').read_text()) if (D/'RESULTS.json').exists() else [];tasks=json.loads((D/'TASKS.json').read_text())
 window_counts['screening']+=sum(r['windows'] for r in rows if r['status']=='experimental_raw_verified');window_counts['isolated']+=sum(r['windows'] for r in rows if r['status']=='formal_raw_verified')
 for i,t in enumerate(tasks):
  screen=[next((r for r in rows if r['index']==i and r['variant']==v and r['attempt']=='new1' and r['status']=='experimental_raw_verified'),None) for v in ['native','fixed']]
  final=next((r for r in rows if r['index']==i and r['attempt']=='final1' and r['status']=='formal_raw_verified'),None)
  if not all(screen) or final is None:pending.append({'batch':name,'index':i,'case':t['case']});continue
  assert final['variant']==min(screen,key=lambda x:x['us'])['variant'];all_rows=[*screen,final];proofs=[];hardware=[]
  for r in all_rows:
   path=D/'remote/runs'/str(i)/r['variant']/r['attempt'];m=json.loads((path/'result.json').read_text());assert sha(path/'result.json')==r['result_sha256'];assert sha(path/'SOURCE.json')==m['source_sha256'];proof=json.loads((path/'SOURCE.json').read_text());proofs.append(proof);hardware.append(m['hardware'])
   for n in ['native_adapters_v2.py','activation_candidate.py']:assert proof['/opt/baseline-over3-20261110/'+name+'/'+n]==sha(D/n)
  common=set.intersection(*(set(x) for x in proofs));assert all(len({x[k] for x in proofs})==1 for k in common)
  for proof in proofs:
   for k in set(proof)-common:assert k.endswith('/rmsnorm.py') or k.endswith('/layernorm_guard.py'),k
  assert len({(x['torch'],x['torch_npu'],x['cube'],x['vector']) for x in hardware})==1
  pairs.append({'key':[t['library'],t['op'],t['case']],'batch':name,'index':i,'selection':final['variant'],'screen_native_us':screen[0]['us'],'screen_fixed_us':screen[1]['us'],'screen_fixed_reduction_pct':100*(1-screen[1]['us']/screen[0]['us']),'selected_isolated_us':final['us'],'isolated_vs_selected_screen_pct':100*(final['us']/min(x['us'] for x in screen)-1),'source_identity_verified':True,'environment':hardware[-1],'evidence':all_rows,'status':'current_environment_selection_verified_historical_ratio_pending'})
out={'pairs':pairs,'pending':pending,'windows':window_counts,'metric':'device latency; HOST_TIMING remains separate diagnostic','historical_ratio_update':False,'caveat':'Current Torch2.13 vs historical2.10; frozen TileOPs denominator has not been realigned. No mixed-protocol or crossenvironment ratio claim.'}
(P/'VLLM_SILU_CURRENT.partial.json').write_text(json.dumps(out,indent=2))
T=P/'delivery/vllm_silu';T.mkdir(exist_ok=True)
for n in ['native_adapters_v2.py','activation_candidate.py']:
 paths=[P/name/n for name in ['vllm-silu-server1']];assert all(f.read_bytes()==paths[0].read_bytes() for f in paths);(T/n).write_bytes(paths[0].read_bytes())
(T/'SELECTION.partial.json').write_text(json.dumps(out,indent=2));(T/'ROUTES.partial.json').write_text(json.dumps([{'key':p['key'],'fixed':p['selection']=='fixed','evidence':'VLLM_SILU_CURRENT.partial.json'} for p in pairs],indent=2))
print(json.dumps({'complete':len(pairs),'pending':len(pending),'windows':window_counts,'pairs':[(p['key'][2],round(p['screen_fixed_reduction_pct'],2),p['selected_isolated_us']) for p in pairs]}))
