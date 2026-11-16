"""Guarded build of the one original FP16 KDA dtype variant needed here.

The generated shell wrapper has a kill-all-recorded-PIDs failure handler. Call
its exact asc_opc command directly instead, retaining every compiler option and
the original generated source/parameter file. All child processes stay in the
reservation's foreground process group.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_partial_native import CACHE,EVIDENCE,main as package


def main():
    from check import checker
    checker.require_guard()
    os.environ['ASCEND_PROCESS_LOG_PATH']=str(EVIDENCE/'native_logs/targeted-kda-build')
    # Publish already finalized original gate/backward variants before the
    # independent KDA compilation, even if this reservation acquires first.
    result=package()
    if result:return result
    build=CACHE/'native-source/build'
    gen=build/'binary/ascend950/gen'
    name='ChunkKdaFwd_1abcbd578c0aecb0d613ba7f77626658'
    target='chunk_kda_fwd_ascend950_0'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    debug=gen/('kernel_meta_'+name)
    # The host guard has already required a quiet device and no other device
    # handles. Preserve our interrupted private compiler directory intact;
    # never inspect, unlink, replace or bypass the host's NPU lock.
    archived=None
    if debug.exists():
        archived=debug.with_name(debug.name+'.preserved-'+stamp)
        debug.rename(archived)
    param=gen/(name+'_param.json')
    source=build/'binary/ascend950/src/chunk_kda_fwd/ChunkKdaFwd.py'
    wrapper=gen/'ChunkKdaFwd-chunk_kda_fwd-0.sh'
    command=['asc_opc',str(source),'--main_func=chunk_kda_fwd',
             '--input_param='+str(param),'--soc_version=Ascend950PR_9599',
             '--output='+str(build/'binary/ascend950/bin/chunk_kda_fwd'),
             '--impl_mode=high_performance,optional','--simplified_key_mode=0','--op_mode=dynamic']
    env=dict(os.environ,ASCEND_CUSTOM_OPP_PATH=str(build/'custom'),HI_PYTHON='python3',
             TILINGKEY_PAR_COMPILE='1',BIN_FILENAME_HASHED='1',
             ASCEND_GLOBAL_LOG_LEVEL='3',ASCEND_SLOG_PRINT_TO_STDOUT='1')
    evidence=dict(schema='fla-original-targeted-build/v1',started_at=stamp,
                  original_cmake_target=target,command=command,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,param,wrapper)},
                  preserved_private_debug_directory=str(archived) if archived else None,
                  process_policy='Same compiler flags; direct foreground invocation avoids stale PID kill handler in generated wrapper.')
    receipt=EVIDENCE/('targeted_kda_build_'+stamp+'.json')
    receipt.write_text(json.dumps(evidence,indent=2)+'\n')
    result=subprocess.run(command,cwd=gen,env=env)
    evidence.update(finished_at=datetime.now(timezone.utc).isoformat(),compiler_returncode=result.returncode)
    receipt.write_text(json.dumps(evidence,indent=2)+'\n')
    if result.returncode:return result.returncode
    for suffix in ('.json','.o'):
        path=build/'binary/ascend950/bin/chunk_kda_fwd'/(name+suffix)
        if not path.is_file():raise RuntimeError('Compiler did not finalize '+str(path))
    packaged=package(CACHE/'python_binary_kda/fla_npu',EVIDENCE/'kda_native_ready.json')
    if packaged:return packaged
    # Verify precisely the compiled dtype variant, using the unchanged full
    # manifest checker in a fresh runtime child under this same reservation.
    # This is deployment validation; the general case queue remains separate.
    output=EVIDENCE/'runs/build-target0-kda-tp1-4x1024-float16.json'
    if not output.exists():
        command=[sys.executable,str(Path(__file__).with_name('launch_case.py')),
                 '--op','KDAFwdOp','--case-id','kda-tp1-4x1024-float16','--output',str(output)]
        evidence.update(phase='checking_compiled_variant',compiled_variant_check_started_at=datetime.now(timezone.utc).isoformat(),compiled_variant_check_command=command,compiled_variant_check_receipt=str(output))
        receipt.write_text(json.dumps(evidence,indent=2)+'\n')
        checked=subprocess.run(command,timeout=300)
        evidence['compiled_variant_check_returncode']=checked.returncode
        evidence['compiled_variant_check_receipt']=str(output)
        receipt.write_text(json.dumps(evidence,indent=2)+'\n')
        if checked.returncode:return checked.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
