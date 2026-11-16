"""Guarded original host build and packaging of completed original kernels."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
CACHE=ROOT/'.cache/all-pairs/fla'
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla'
SOURCE=ROOT/'ascend_baselines/flash-linear-attention-npu'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(target=None,receipt=None,host_receipt=None):
    # Reuse the original checker guard assertion before any CANN build step.
    if host_receipt is None:
        from check import checker
        checker.require_guard()
    os.environ.setdefault('ASCEND_PROCESS_LOG_PATH',str(EVIDENCE/'native_logs/partial-package'))
    target=target or CACHE/'python_binary_partial/fla_npu'
    receipt=receipt or EVIDENCE/'partial_native_ready.json'
    if receipt.exists() and target.exists():return 0
    if target.exists():raise FileExistsError('Preserve unfinished deployment for inspection: '+str(target))
    build=CACHE/'native-source/build'
    build_env=dict(os.environ)
    build_env['LD_LIBRARY_PATH']=str(build)+os.pathsep+build_env.get('LD_LIBRARY_PATH','')
    if host_receipt is None:
        subprocess.run(['cmake','--build','.', '--target','cust_proto','cust_opapi','cust_opmaster','-j2'],cwd=build,env=build_env,check=True)
    else:
        # A separately audited CPU-only exact g++ build already finalized the
        # host libraries. This path only reads/copies files and runs the
        # original stdlib JSON configuration generator, without importing CANN.
        host=json.loads(Path(host_receipt).read_text())
        if host.get('schema')!='fla-host-only-finalization/v1' or host.get('strace_device_open_hits')!=[]:
            raise RuntimeError('Missing audited CPU host build receipt')
        for name,digest in host['source_sha256'].items() | host['artifact_sha256'].items():
            if sha(Path(name))!=digest:raise RuntimeError('CPU host build source/artifact drift: '+name)
    api=build/'libcust_opapi.so'
    exports=subprocess.check_output(['nm','-D','--defined-only',str(api)],text=True)
    symbols={line.split()[-1] for line in exports.splitlines() if line.split()}
    required=['aclnnKdaGateCumsum','aclnnChunkGatedDeltaRuleBwd','aclnnChunkGatedDeltaRuleFwdPrepare']
    for name in required:
        for suffix in ('','GetWorkspaceSize'):
            if name+suffix not in symbols:raise RuntimeError('Missing original export: '+name+suffix)
    shutil.copytree(SOURCE/'torch_custom/fla_npu/fla_npu',target,ignore=shutil.ignore_patterns('opp','__pycache__','*.pyc'))
    vendor=target/'opp/vendors/fla_npu_transformer'
    lineage={}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(src)!=sha(dst):raise RuntimeError('Copy mismatch: '+str(dst))
        lineage[str(dst.relative_to(ROOT))]=dict(source=str(src.relative_to(ROOT)),sha256=sha(src))
    for name,relative in [('libcust_opapi.so','op_api/lib'),('libopapi_math.so','op_api/lib'),('libcust_opsproto_rt2.0.so','op_proto/lib/linux/x86_64'),('libcust_opmaster_rt2.0.so','op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64')]:
        copy(build/name,vendor/relative/name)
    tiling=vendor/'op_impl/ai_core/tbe/op_tiling'
    (tiling/'liboptiling.so').symlink_to('lib/linux/x86_64/libcust_opmaster_rt2.0.so')
    kernels=vendor/'op_impl/ai_core/tbe/kernel/ascend950'
    complete=[]
    # Only finalized object+metadata pairs; never ship compiler temporary files.
    for metadata in sorted((build/'binary/ascend950/bin').glob('*/*.json')):
        binary=metadata.with_suffix('.o')
        if not binary.is_file():continue
        subprocess.run(['readelf','--file-header','--section-headers',str(binary)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        relative=metadata.relative_to(build/'binary/ascend950/bin')
        copy(binary,(kernels/relative).with_suffix('.o'))
        copy(metadata,kernels/relative)
        complete.append(str(relative))
    if not any(name.startswith('kda_gate_cumsum/') for name in complete):raise RuntimeError('No completed original gate kernels')
    utility=SOURCE/'cmake/scripts/util/ascendc_ops_config.py'
    config=vendor/'op_impl/ai_core/tbe/kernel/config/ascend950'
    # The unchanged original utility only imports Python stdlib/const_var. It
    # updates deployment metadata filePath fields on these private copies.
    command=[sys.executable,str(utility),'-p',str(kernels),'-s','ascend950','-o',str(config)]
    if host_receipt is not None:
        argv=command[1:]
        bootstrap='import sys,runpy;sys.path.insert(0,'+repr(str(utility.parent))+');sys.argv='+repr(argv)+';runpy.run_path('+repr(str(utility))+',run_name="__main__")'
        command=[sys.executable,'-I','-S','-B','-c',bootstrap]
    subprocess.run(command,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0'))
    for source in (build/'impl/dynamic').glob('*.py'):
        copy(source,vendor/'op_impl/ai_core/tbe/fla_npu_transformer_impl/dynamic'/source.name)
    ops_info=build/'autogen/aic-ascend950-ops-info.json'
    copy(ops_info,vendor/'op_impl/ai_core/tbe/config/ascend950'/ops_info.name)
    python_hashes={str(p.relative_to(ROOT)):sha(p) for p in target.rglob('*.py') if '/opp/' not in str(p)}
    artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in vendor.rglob('*') if p.is_file()}
    value=dict(schema='fla-partial-original-deployment/v1',updated_at=datetime.now(timezone.utc).isoformat(),deployment=str(target),original_defined_symbols=required,completed_kernel_metadata=complete,source_lineage=lineage,python_sha256=python_hashes,artifact_sha256=artifact_hashes,configuration_generator=dict(file=str(utility.relative_to(ROOT)),sha256=sha(utility)),limitations='Only finalized kernel variants are deployed. ChunkKdaFwd remains unavailable unless separately completed; no device validation is implied by this packaging receipt.')
    if host_receipt is not None:
        value['cpu_host_build_receipt']=dict(path=str(host_receipt),sha256=sha(Path(host_receipt)))
    for name,item in lineage.items():
        item['published_sha256']=artifact_hashes[name]
        if item['published_sha256']!=item['sha256']:
            item['transformation']='Original ascendc_ops_config.py deployment metadata rewrite; native binary bytes unchanged'
    temporary=receipt.with_suffix(receipt.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(receipt)
    print(json.dumps(dict(deployment=str(target),completed_variants=len(complete))),flush=True)
    return 0


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--host-receipt',type=Path)
    args=parser.parse_args()
    raise SystemExit(main(host_receipt=args.host_receipt))
