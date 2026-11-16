from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;D=P/'vllm-rope-dual-server2'
rs=json.loads((D/'RESULTS.json').read_text()) if (D/'RESULTS.json').exists() else [];out=[]
for r in rs:
 if r['variant']!='fixed' or r['status']!='formal_raw_verified':continue
 c=next(x for x in rs if x['index']==r['index'] and x['variant']=='native' and x['status']=='formal_raw_verified')
 proofs=[]
 for x in (c,r):
  f=D/'remote/runs'/str(x['index'])/x['variant']/x['attempt']/'result.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==x['result_sha256'];proofs.append(json.loads((f.parent/'SOURCE.json').read_text()))
 assert proofs[0]==proofs[1]
 out.append(dict(index=r['index'],case=r['case'],original_host_us=c['host_us'],candidate_host_us=r['host_us'],original_device_us=c['us'],candidate_device_us=r['us'],host_reduction_fraction=1-r['host_us']/c['host_us'],device_change_fraction=r['us']/c['us']-1,original=c,candidate=r))
f=P/'VLLM_ROPE_DUAL_METRIC.partial.json'
f.write_text(json.dumps({'status':'same-run host/device audited; final historical speedup selection pending metric provenance','pairs':out,'formal_windows':30*len(out)},indent=2)+'\n')
print([(r['case'],round(r['host_reduction_fraction']*100,1),round(r['device_change_fraction']*100,1)) for r in out])
