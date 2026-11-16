from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from ops_nn_adalnv2_cached_fidelity import validate_source
i=int(sys.argv[1]);d=P/'prepared-rollout'/str(i);m=json.loads((d/'prepare.json').read_text());validate_source(P,i,m)
for name,key in [('payload.pt','payload_sha256'),('case.pkl','descriptor_sha256')]:assert hashlib.sha256((d/name).read_bytes()).hexdigest()==m[key]
print('CPU_ORIGINAL_CACHED_ADALNV2_SOURCE_PARAMS_PAYLOAD_VERIFIED',i,flush=True)
