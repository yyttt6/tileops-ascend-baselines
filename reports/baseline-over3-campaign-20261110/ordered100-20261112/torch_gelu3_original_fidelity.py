"""Fresh exact-publication source/input identity; no aliases or metadata rewriting."""
from pathlib import Path
import json,hashlib,datetime,sys
APPROVED={1361,1362,1363}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_runtime_source_identity(P,i,meta):
 P=Path(P);assert i in APPROVED and meta['source_commit']=='f037318f1483a8e4b17c30989e86debf0751a489' and meta['key'][0:2]==['torch-eager','GELUAndMulFwdOp'];assert meta['params']['approximate']=='tanh' and meta['measured_entry']=='case.torch (original storage-dtype torch_program)' and meta['no_NPU_imported'];assert all(sha(p)==h for p,h in meta['sources'].items());d=P/'prepared-torch-gelu3-f037'/str(i);assert sha(d/'payload.pt')==meta['payload_sha256'] and sha(d/'case.pkl')==meta['descriptor_sha256'];tag=sys.argv[1];out=P/'runs'/tag/str(i);out.mkdir(parents=True,exist_ok=True);(out/'cached_input_source_proof.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':i,'source_commit':meta['source_commit'],'prepare_sha256':sha(d/'prepare.json'),'payload_sha256':meta['payload_sha256'],'no_source_delta':True,'original_approximate_and_storage_Torch_entry_preserved':True,'original_checker_unchanged':True},indent=2)+'\n');return True
