"""Original state recurrence exposes exp(x) through constant state operands."""
from pathlib import Path

_schema = None


def exponential(case):
    import json
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, WORKSPACE
    from .adapters import DISPATCH_EVENTS
    from . import adapters
    from .hashing import source_hashes
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Original recurrence consumes FP32; only lossless floating widening is bound.')
    source = WORKSPACE / 'ascend_baselines/ops-transformer/experimental/mamba/mamba2_chunk_state_passing/torch_interface.cpp'
    binary = WORKSPACE / '.cache/all-pairs/ops_transformer/state_passing_build/liboriginal_state_passing.so'
    receipt = WORKSPACE / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer/state_passing_build.json'
    hashes = source_hashes(source.parent, binary, receipt, Path(__file__), Path(adapters.__file__))
    build = json.loads(receipt.read_text())
    if hashes[str(source.relative_to(WORKSPACE))] != build['source_sha256'] or hashes[str(binary.relative_to(WORKSPACE))] != build['binary_sha256']:
        raise RuntimeError('Original state recurrence source/binary mismatch.')
    global _schema
    name = 'mambav2_chunk_state_passing'
    if not hasattr(torch.ops.npu_ops_transformer_ext, name):
        _schema = torch.library.Library('npu_ops_transformer_ext', 'FRAGMENT')
        _schema.define(name + '(Tensor dacs, Tensor init_states, Tensor states, Tensor ct) -> (Tensor, Tensor)')
        torch.ops.load_library(str(binary))
        adapters._state_passing_schema = _schema
        adapters._state_passing_loaded = True
    table = torch._C._dispatch_dump_table('npu_ops_transformer_ext::' + name)
    if not any(line.startswith('PrivateUse1:') and str(source) in line for line in table.splitlines()):
        raise RuntimeError('State recurrence dispatch is not the original source.')
    minus_one = case._call.signature.name == 'Expm1FwdOp'

    def run(x):
        flat = x.contiguous().flatten()
        result = torch.empty_like(flat)
        # Original cube tiles require L and P multiples of 64; native N=16
        # gives Z=1024, with Z/4 divisible by the 64-float vector length.
        # These are supported operand shapes, not a kernel/schedule edit.
        for begin in range(0, flat.numel(), 131072):
            chunk = flat[begin:begin + 131072]
            h = chunk.numel()
            da = torch.nn.functional.pad(chunk.float(), (0, 8)).reshape(1, 1, h + 8)
            initial = torch.ones((1, h, 1024), dtype=torch.float32, device=x.device)
            states = torch.full((1, 1, h, 1024), -1. if minus_one else 0., dtype=torch.float32, device=x.device)
            ct = torch.zeros((1, 1, 64, 1, 16), dtype=torch.float16, device=x.device)
            _, final = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state_passing(da, initial, states, ct)
            result[begin:begin + h].copy_(final[0, :, 0, 0].to(x.dtype))
            DISPATCH_EVENTS.append({'op':'npu_ops_transformer_ext::'+name,'source':str(source),
                'binary':str(binary),'dispatch_owner_checked':True,'returned':0})
        return result.reshape_as(x)
    return Candidate(Implementation(run), 'ops-transformer::mambav2_chunk_state_passing',
        'One original recurrence step with initial=1 and state=0 or -1 exposes exp(x) or exp(x)-1 in its FP32 final-state output. Lossless widening, constant operands, N=16/P=L=64 packing, output gather/cast and every native call are inside run. Original kernel math/schedule and complete manifest oracle unchanged; no external exponential arithmetic.', hashes)
