"""CPU-only copy/audit of an already completed original CPack staging tree."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[4]
CACHE=ROOT/'.cache/all-pairs/fla'
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla'
VENDOR=ROOT/'ascend_baselines/flash-linear-attention-npu'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build=CACHE/'native-source/build'
    archive=build/'fla_npu_linux-x86_64.run'
    staging=build/'_CPack_Packages/Linux/External/fla_npu_linux-x86_64.run'
    if not archive.is_file():
        raise RuntimeError('The original CPack package is not complete: '+str(archive))
    candidates=list(staging.rglob('vendors/fla_npu_transformer'))
    if len(candidates)!=1:
        raise RuntimeError('Expected one completed original vendor staging directory: '+repr(candidates))
    original=candidates[0]
    binary=original/'op_api/lib/libcust_opapi.so'
    defined=subprocess.check_output(['nm','-D','--defined-only',str(binary)],text=True)
    symbols={line.split()[-1] for line in defined.splitlines() if line.split()}
    required=['aclnnChunkKdaFwd','aclnnChunkGatedDeltaRuleBwd','aclnnChunkGatedDeltaRuleFwdPrepare','aclnnKdaGateCumsum']
    missing=[name+suffix for name in required for suffix in ('','GetWorkspaceSize') if name+suffix not in symbols]
    if missing:raise RuntimeError('Required owned export definitions missing: '+repr(missing))
    target=CACHE/'python_binary/fla_npu'
    if target.exists():
        raise FileExistsError('Preserve existing deployment; do not overwrite: '+str(target))
    source=VENDOR/'torch_custom/fla_npu/fla_npu'
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('opp','__pycache__','*.pyc'))
    shutil.copytree(original,target/'opp/vendors/fla_npu_transformer',symlinks=True)
    python_hashes={}
    for path in source.rglob('*.py'):
        deployed=target/path.relative_to(source)
        if path.read_bytes()!=deployed.read_bytes():raise RuntimeError('Python source mismatch: '+str(deployed))
        python_hashes[str(deployed.relative_to(ROOT))]=digest(deployed)
    installed=target/'opp/vendors/fla_npu_transformer'
    artifacts={}
    for path in original.rglob('*'):
        if not path.is_file():continue
        copied=installed/path.relative_to(original)
        if digest(path)!=digest(copied):raise RuntimeError('CPack artifact mismatch: '+str(copied))
        artifacts[str(copied.relative_to(ROOT))]=digest(copied)
    receipt=dict(schema='fla-original-native-deployment/v1',updated_at=datetime.now(timezone.utc).isoformat(),
                 package=str(archive),package_sha256=digest(archive),staging=str(original),deployment=str(target),
                 required_defined_symbols=required,python_sha256=python_hashes,artifact_sha256=artifacts,
                 source_snapshot=str(EVIDENCE/'native_source_snapshot_complete.json'),
                 device_validation='Pending; this CPU-only step proves byte lineage and owned export definitions, not execution.')
    (EVIDENCE/'native_new_deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(deployment=str(target),python_files=len(python_hashes),artifacts=len(artifacts))))


if __name__=='__main__':main()
