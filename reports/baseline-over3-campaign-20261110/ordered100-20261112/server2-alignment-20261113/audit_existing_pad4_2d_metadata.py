"""Audit successful original metadata generation and immutable16 artifacts; no compiler/generator rerun."""
from pathlib import Path
import json,hashlib,datetime,os
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1');old=N.parent/'ops-nn-unary-gemm10-v1';src=old/'source';b=src/'build-device-v1/binary/ascend950/bin';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['child']['pid']==os.getpgrp();plan=json.loads((N/'PAD4_COMPILE_SELECTION_H1.json').read_text());ledger=json.loads((N/'PAD4_COMPILE_RESUME_LEDGER.json').read_text());sm=json.loads((old/'SOURCE_MANIFEST.json').read_text());assert all(sha(src/k)==v['sha256'] for k,v in sm['files'].items());assert len(ledger['compiled'])==16
for h,r in ledger['compiled'].items():
 assert h==r['command']['command_sha256'] and sha(r['command']['param'])==r['command']['param_sha256']
 for p,v in r['artifacts'].items():assert sha(p)==v['sha256'],p
cfg=json.loads((b/'config/ascend950/binary_info_config.json').read_text());assert {k:len(v['binaryList']) for k,v in cfg.items()}=={'BatchMatMulV3':6,'FusedMatMul':2,'MatMulV3':8};rawobjects={str(Path(p).relative_to(b)):v['sha256'] for r in ledger['compiled'].values() for p,v in r['artifacts'].items() if p.endswith('.o')}
for v in cfg.values():
 for row in v['binaryList']:
  for key in ['binPath','jsonPath']:assert (b/row[key]).is_file(),row[key]
  assert row['binPath'] in rawobjects and sha(b/row['binPath'])==rawobjects[row['binPath']]
selected=[r for r in plan['selected_whole_commands'] if r['op']=='MatMulV3'];assert len(selected)==8 and all(r['command_sha256'] in ledger['compiled'] for r in selected);objects={p:h for p,h in rawobjects.items() if Path(p).name.startswith('MatMulV3_')};assert len(objects)==8;params={r['param'] for r in selected}
import shlex
out={**plan,'selected_whole_commands':selected,'selected_original_out_cmd':[c for c in plan['selected_original_out_cmd'] if shlex.split(c)[2] in params],'actual_objects':objects,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original18_selection_sha256':sha(N/'PAD4_COMPILE_SELECTION_H1.json'),'resume_ledger_sha256':sha(N/'PAD4_COMPILE_RESUME_LEDGER.json'),'only_frozen2D_API_scope':[490,494,496,497],'dependency_scope_review_sha256':sha(N/'PAD4_2D_DEPENDENCY_SCOPE_REVIEW.json'),'original_metadata_retains16_completeobjects_unchanged':rawobjects,'held_unrelated_BF16_Fused_not_part_of_2D_dispatch':True,'no_compile_generator_kernel_flags_or_dispatch_edits':True};p=N/'PAD4_2D_COMPILE_COMPLETED_H1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('Originalmetadata16objects allSHAvalid;8MatMulV3 complete2Dscope; no compile/generator rerun',flush=True)
