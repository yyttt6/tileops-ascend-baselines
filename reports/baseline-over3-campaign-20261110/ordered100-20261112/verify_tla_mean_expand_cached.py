"""CPU-only finite cached-input qualification; no metadata rewrite."""
import pathlib,json,sys,datetime
from tla_mean_expand_cached_fidelity import APPROVED,validate_source,sha
P=pathlib.Path(__file__).resolve().parent;i=int(sys.argv[1]);assert i in APPROVED,'explicit scope only'
D=P/'prepared-rollout'/str(i);m=json.loads((D/'prepare.json').read_text());proof=validate_source(P,i,m)
assert sha(D/'payload.pt')==m['payload_sha256'] and sha(D/'case.pkl')==m['descriptor_sha256']
proof.update({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'CPU_full_payload_and_source','no_NPU_imported':True})
out=P/'cached-input-fidelity'/str(i);out.mkdir(parents=True,exist_ok=True);(out/'PROOF.json').write_text(json.dumps(proof,indent=2));print('FROZEN_OPS_NN_UNARY_CACHED_INPUT_VERIFIED',i,proof['prepare_sha256'],flush=True)
