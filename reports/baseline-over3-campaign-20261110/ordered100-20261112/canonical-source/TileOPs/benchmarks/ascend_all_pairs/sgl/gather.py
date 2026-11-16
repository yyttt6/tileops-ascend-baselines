"""Original indexed-copy kernel with isolated A5 deployment compatibility."""
from pathlib import Path


def build(case):
    import torch
    import subprocess
    import hashlib
    import json
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes,LIBRARIES
    dst,cache,table,cu,starts,scale=case.inputs
    from .additional import mul
    multiply=mul(case) if scale is not None else None
    root=Path(__file__).resolve().parents[4]
    directory=root/'.cache/all-pairs/sgl/unidex'
    directory.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('native_unidex')
    binary=directory/'libsgl_unidex_compat.so'
    build_sources=[source/'CMakeLists.txt',source/'binding.cpp',
        LIBRARIES/'sgl-kernel-npu/csrc/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp',
        LIBRARIES/'sgl-kernel-npu/csrc/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp']
    fingerprint=hashlib.sha256(b''.join(f.read_bytes() for f in build_sources)).hexdigest()
    previous=directory/'last_build_failure.json'
    success=directory/'build_success.json'
    compatible=binary.exists() and success.exists() and json.loads(success.read_text()).get('source_fingerprint')==fingerprint
    if not compatible:
        if previous.exists() and json.loads(previous.read_text())['source_fingerprint']==fingerprint:
            raise RuntimeError(f'Identical original unidex build already failed; no redundant compile retry. Evidence: {directory / "build.log"}')
        # Factory invocation is guarded with the entire import/build/run process.
        try:
            with (directory/'build.log').open('a') as log:
                subprocess.run(['cmake','-S',str(source),'-B',str(directory)],check=True,stdout=log,stderr=subprocess.STDOUT)
                subprocess.run(['cmake','--build',str(directory),'-j','2'],check=True,stdout=log,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            previous.write_text(json.dumps({'source_fingerprint':fingerprint,'command':exc.cmd,'exit_code':exc.returncode,'log':str(directory/'build.log')},indent=2)+'\n')
            raise
        success.write_text(json.dumps({'source_fingerprint':fingerprint,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2)+'\n')
    torch.ops.load_library(str(binary))
    def run(dst,cache,table,cu,starts=None,scale=None):
        offsets=cu.cpu().tolist()
        bases=[0]*(len(offsets)-1) if starts is None else starts.cpu().tolist()
        source_indices=[]
        for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
            positions=torch.arange(bases[i],bases[i]+end-begin,device=cache.device,dtype=torch.int64)
            source_indices.append(table[i,positions//cache.shape[1]].long()*cache.shape[1]+positions%cache.shape[1])
        src_index=torch.cat(source_indices).contiguous()
        output=torch.empty_like(dst)
        dest_index=torch.arange(dst.shape[0],device=dst.device,dtype=torch.int64)
        valid=torch.ones(dst.shape[0],device=dst.device,dtype=torch.bool)
        payload=cache
        if scale is not None:payload=multiply.implementation.run(cache.float(),scale.float())
        payload=payload.to(dst.dtype).contiguous()
        torch.ops.allpairs_sgl_unidex.copy(payload,output,src_index,dest_index,valid,
            cache.shape[0]*cache.shape[1],dst.shape[0],dst.numel()//dst.shape[0]*dst.element_size(),dst.shape[0])
        return output
    vendor=LIBRARIES/'sgl-kernel-npu'
    hashes=source_hashes(Path(__file__),source/'CMakeLists.txt',source/'binding.cpp',binary,
            vendor/'csrc/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp',
            vendor/'csrc/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp',
            vendor/'csrc/utils/torch_helper.h',vendor/'csrc/utils/defines.h',vendor/'csrc/utils/cann_graph_compat.h',
            vendor/'cmake/config_envs.cmake',vendor/'cmake/config_ascend.cmake')
    if multiply is not None:hashes.update(multiply.source_sha256)
    return Candidate(Implementation(run),'sgl_kernel_npu::unidex_copy (original host/kernel, isolated A5 registration) + optional native scale_shift',
        'Original source compiled unchanged for the installed A5 SDK and exposed through a registration-only bridge. Map actual page/sequence metadata to native source rows. Optional lossless FP8 widening, original native scale multiplication and declared output cast precede native indexed byte copy; all transformations are per-call. No Torch gather computes cache payloads.',
        hashes,timing_mode='host_roundtrip')
