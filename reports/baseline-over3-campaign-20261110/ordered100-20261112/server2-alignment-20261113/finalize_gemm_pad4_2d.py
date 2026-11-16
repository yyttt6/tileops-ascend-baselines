"""Only complete original MatMulV3 commands needed by proven 2D CAPI; no runtime branch edits."""
from pathlib import Path
import json,hashlib,subprocess,shlex,datetime,os
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1');old=N.parent/'ops-nn-unary-gemm10-v1';src=old/'source';b=src/'build-device-v1/binary/ascend950/bin';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['child']['pid']==os.getpgrp();plan=json.loads((N/'PAD4_COMPILE_SELECTION_H1.json').read_text());ledger=json.loads((N/'PAD4_COMPILE_RESUME_LEDGER.json').read_text());source=json.loads((old/'SOURCE_MANIFEST.json').read_text());assert all(sha(src/k)==v['sha256'] for k,v in source['files'].items());selected=[v for v in plan['selected_whole_commands'] if v['op']=='MatMulV3'];assert len(selected)==8;objects={}
for row in selected:
 assert row['command_sha256'] in ledger['compiled'] and sha(row['param'])==row['param_sha256']
 for p,v in ledger['compiled'][row['command_sha256']]['artifacts'].items():
  assert sha(p)==v['sha256']
  if p.endswith('.o'):objects[str(Path(p).relative_to(b))]=v['sha256']
params={v['param'] for v in selected};commands=[v for v in plan['selected_original_out_cmd'] if shlex.split(v)[2] in params];assert len(commands)==8
for cmd in commands:subprocess.run(shlex.split(cmd),cwd=src/'scripts/kernel/binary_script',check=True)
subprocess.run(['cmake','--build',str(src/'build-device-v1'),'--target','gen_bin_info_config','--parallel','1'],check=True);cfg=json.loads((b/'config/ascend950/binary_info_config.json').read_text());assert set(cfg)=={'MatMulV3'},list(cfg)
for v in cfg['MatMulV3']['binaryList']:
 for key in ['binPath','jsonPath']:assert (b/v[key]).is_file() or (b/'ascend950'/Path(v[key]).relative_to('ascend950')).is_file()
out={**plan,'selected_whole_commands':selected,'selected_original_out_cmd':commands,'actual_objects':objects,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_flag_or_JSON_or_kernel_source_edits':True,'original18_selection_sha256':sha(N/'PAD4_COMPILE_SELECTION_H1.json'),'resume_ledger_sha256':sha(N/'PAD4_COMPILE_RESUME_LEDGER.json'),'only_frozen2D_API_scope':[490,494,496,497],'dependency_scope_review_sha256':sha(N/'PAD4_2D_DEPENDENCY_SCOPE_REVIEW.json'),'unrelated_FusedBF16_compile_failure_held':True};(N/'PAD4_2D_COMPILE_COMPLETED_H1.json').write_text(json.dumps(out,indent=2)+'\n');print('Original8MatMulV3 commands alreadycompiled, originalmetadata finalized, no kernelcompile rerun',flush=True)
