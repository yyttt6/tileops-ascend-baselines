"""Original ops-transformer chunk-state binary with layout-only bindings."""
import json
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes

_schema = None
_loaded = False


def transformer_ssd(case):
    # Preserve the historical implementation below for provenance; do not allow
    # a small-amplitude absolute-tolerance pass to re-enable this native entry.
    quarantine_reason = 'Original ops-transformer Mamba SSD entry is quarantined on Ascend950: large output error and failed changed-input CPU check despite a historical loose-tolerance pass. See CLASSIC_BASELINE_PILOT/runs/classics_190/root_review.json. Kernel/schedule repair is outside this adapter scope.'
    if quarantine_reason:
        raise RuntimeError(quarantine_reason)
    import torch
    from benchmarks.api import Implementation

    global _schema, _loaded
    root = LIBRARIES / 'ops-transformer'
    cache = WORKSPACE / '.cache/ascend-classic/native-pilot-20261020/ops-transformer-mamba'
    receipt_path = cache / 'build_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    binary = cache / 'libclassic_mamba_original.so'
    files = [WORKSPACE / name for name in receipt['original_source_sha256']]
    actual = source_hashes(*files, binary)
    for name, digest in receipt['original_source_sha256'].items():
        if actual[name] != digest:
            raise RuntimeError('Original Mamba source changed since the isolated build: ' + name)
    if actual[str(binary.relative_to(WORKSPACE))] != receipt['binary_sha256']:
        raise RuntimeError('Original Mamba binary changed since the isolated build.')
    schema_file = root / 'experimental/npu_ops_transformer_ext/npu_ops_transformer_ext/npu_ops_def.cpp'
    schema = 'mambav2_chunk_state(Tensor dt_out, Tensor dacs, Tensor bt, Tensor xt) -> Tensor'
    if 'm.def("' + schema + '");' not in schema_file.read_text():
        raise RuntimeError('The original Mamba API schema no longer matches.')
    if not _loaded:
        if hasattr(torch.ops.npu_ops_transformer_ext, 'mambav2_chunk_state'):
            raise RuntimeError('A different Mamba registration was loaded before this isolated binary.')
        _schema = torch.library.Library('npu_ops_transformer_ext', 'FRAGMENT')
        _schema.define(schema)
        torch.ops.load_library(str(binary))
        table = torch._C._dispatch_dump_table('npu_ops_transformer_ext::mambav2_chunk_state')
        owner = str(root / 'experimental/mamba/mamba2_chunk_state/torch_interface.cpp')
        if not any(line.startswith('PrivateUse1:') and owner in line for line in table.splitlines()):
            raise RuntimeError('Mamba PrivateUse1 binding did not resolve to its original native source.')
        _loaded = True
    fn = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state

    def run(x, bmat, dt, da, seq_idx):
        batch, length, heads, dim = x.shape
        _, dt_heads, chunks, chunk_len = dt.shape
        groups, states = bmat.shape[2:]
        if (dt_heads != heads or length != chunks * chunk_len or dim != 64
                or heads % 8 or states % 64 or chunk_len % 128
                or heads % groups or (heads // groups) % 8):
            raise UnsupportedCase('Original Mamba tiles require P=64, H and heads/group divisible by8, N by64, chunk length by128.')
        if dt.dtype != torch.float32 or da.dtype != torch.float32:
            raise UnsupportedCase('Original dt/dA inputs must retain FP32.')
        # The original native exponential omits the reference clamp. Validate
        # that the clamp is neutral; never replace it with new kernel arithmetic.
        if not bool((da[..., -1:] <= da).all().item()):
            raise UnsupportedCase('Original Mamba requires end-of-chunk decay no greater than earlier cumulative decay.')
        if seq_idx is not None:
            ids = seq_idx.reshape(batch, chunks, chunk_len)
            # All current manifest sequence boundaries fall between chunks.
            # Validate supplied metadata on every invocation before omitting it.
            if not bool(((ids == ids[..., -1:]) & (ids >= 0)).all().item()):
                raise UnsupportedCase('Original Mamba has no within-chunk sequence masking API.')
        native = fn(dt.permute(0, 2, 3, 1).contiguous(),
                    da.permute(0, 2, 3, 1).contiguous(),
                    bmat.reshape(batch, chunks, chunk_len, groups, states).contiguous(),
                    x.reshape(batch, chunks, chunk_len, heads, dim).contiguous())
        return native.transpose(-1, -2).contiguous()

    return Candidate(
        Implementation(run), 'ops-transformer::mambav2_chunk_state',
        'Original complete native FP32-dt/dA chunk-state entry; 950 build target and dependency linking only. Reshape sequence into chunks, transpose dt/dA and output; validate chunk-aligned sequence metadata and neutral decay clamp. Original native FP16 conversion, tiles, buffering and synchronization unchanged.',
        source_hashes(Path(__file__), receipt_path, schema_file) | actual,
        limitations=('Full CPU manifest reference, unchanged tolerance and input preservation are required; original native API converts x/B to FP16.',),
    )
