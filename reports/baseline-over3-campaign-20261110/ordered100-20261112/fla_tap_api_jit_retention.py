from pathlib import Path
import json,hashlib
from fla_tap_triton_identity import capture as tap_capture
from fla_cube_api_jit_retention import capture as aot_capture
def capture(work,destination,require_aot=False,aot_kind="chunk"):
 r=tap_capture(work,destination)
 if require_aot:
  out=Path(destination)
  if aot_kind=='gate':
   from fla_gate_api_aot_retention import capture as selected
  else:selected=aot_capture
  selected(work,out/'AOT');r['aot_retention_sha256']=hashlib.sha256((out/'AOT/RETENTION.json').read_bytes()).hexdigest();r['actual_Triton_and_original_AoT_identity_required']=True;(out/'RETENTION.json').write_text(json.dumps(r,indent=2)+chr(10))
 return r
