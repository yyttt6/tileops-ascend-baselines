from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-state-original-v1')
DISPATCH_EVENTS=[]
_state_passing_schema=None
_state_passing_loaded=False
def source_hashes(*paths):
 out={}
 for p in paths:
  p=Path(p)
  for f in ([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()]):out[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return out
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='SSDStatePassingFwdOp'
  return state_passing(case)
 return build
def state_passing(case):
    import json, torch
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, WORKSPACE, source_hashes
    from ops_transformer_state_api_support import source_hashes
    from benchmarks.api import Implementation
    if case.inputs[0].shape[-1] > 8192:
        raise UnsupportedCase('Original native state vector is padded to supported N=128,P=64; larger vectors require a separately checked original tiling.')
    global _state_passing_schema, _state_passing_loaded
    source = N / 'source/experimental/mamba/mamba2_chunk_state_passing/torch_interface.cpp'
    binary = N / 'liboriginal_state_passing.so'
    receipt = N / 'BUILD_RECIPE_RESULT.json'
    hashes = source_hashes(source.parent, binary, receipt, Path(__file__))
    build = json.loads(receipt.read_text())
    if hashes[str(source)] != build['source_sha256'] or hashes[str(binary)] != build['binary_sha256']:
        raise RuntimeError('Original state-passing source/binary mismatch.')
    if not _state_passing_loaded:
        if hasattr(torch.ops.npu_ops_transformer_ext, 'mambav2_chunk_state_passing'):
            raise RuntimeError('Another state-passing implementation is already registered.')
        _state_passing_schema = torch.library.Library('npu_ops_transformer_ext', 'FRAGMENT')
        _state_passing_schema.define('mambav2_chunk_state_passing(Tensor dacs, Tensor init_states, Tensor states, Tensor ct) -> (Tensor, Tensor)')
        torch.ops.load_library(str(binary))
        table = torch._C._dispatch_dump_table('npu_ops_transformer_ext::mambav2_chunk_state_passing')
        if not any((line.startswith('PrivateUse1:') and str(source) in line for line in table.splitlines())):
            raise RuntimeError('State-passing dispatch is not original source.')
        _state_passing_loaded = True

    def run(states, da, initial):
        b, c, h, d = states.shape
        state = torch.zeros((b, h, 8192), device=states.device, dtype=torch.float32) if initial is None else torch.nn.functional.pad(initial.float(), (0, 8192 - d)).contiguous()
        packed = torch.nn.functional.pad(states.float(), (0, 8192 - d)).contiguous()
        ct = torch.zeros((b, 1, 64, 1, 128), dtype=torch.float16, device=states.device)
        previous = []
        for chunk in range(c):
            previous.append(state[..., :d].clone())
            native_da = da[:, :, chunk:chunk + 1].transpose(1, 2).clone(memory_format=torch.contiguous_format)
            native_chunk = packed[:, chunk:chunk + 1].clone(memory_format=torch.contiguous_format)
            _, final = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state_passing(native_da, state, native_chunk, ct)
            state = final.reshape(b, h, 8192)
            DISPATCH_EVENTS.append({'op': 'npu_ops_transformer_ext::mambav2_chunk_state_passing', 'source': str(source), 'binary': str(binary), 'dispatch_owner_checked': True, 'returned': 0})
        return (torch.stack(previous, 1), state[..., :d].contiguous())
    return Candidate(Implementation(run), 'ops-transformer::mambav2_chunk_state_passing', 'Original FP32 state recurrence, one original API call per chunk with zero-padded state. Select FP32 final state and carry it to the next original call; collect complete previous states without arithmetic. Native half-projection output is unused. Original source/math/schedule unchanged.', hashes)
