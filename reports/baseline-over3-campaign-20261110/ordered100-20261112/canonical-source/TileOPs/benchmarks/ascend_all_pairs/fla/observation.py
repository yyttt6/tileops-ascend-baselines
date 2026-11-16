"""Read-only evidence for original checker execution; no input substitutions."""
import hashlib
from pathlib import Path


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        # Capture logical bytes from the actual device value, never generate data.
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                    device=str(value.device), numel=value.numel(), sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value, (tuple, list)):
        return [describe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    return value


def triton_return(frame, kernel, evidence):
    """JITFunction.run returns the exact compiled object after a real launch."""
    if kernel is None or frame.f_locals.get('warmup', True):
        return None
    fn = getattr(frame.f_locals.get('self'), 'fn', None)
    if fn is None or not hasattr(fn, '__code__'):
        return None
    source = Path(fn.__code__.co_filename).resolve()
    if 'flash-linear-attention-npu' not in str(source):
        return None
    binary = getattr(kernel, 'kernel', None)
    if not isinstance(binary, bytes):
        return dict(jit_function=fn.__name__, error='Returned kernel has no binary bytes', execution_scope='candidate')
    digest = hashlib.sha256(binary).hexdigest()
    path = evidence / 'native_objects' / (digest + '.bin')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(binary)
    return dict(jit_function=fn.__name__, jit_source=str(source),
                jit_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                binary_sha256=digest, binary_bytes=len(binary), binary_snapshot=str(path),
                compiled_hash=str(getattr(kernel, 'hash', '')), kernel_name=getattr(kernel, 'name', None),
                runtime_return_observed=True, warmup=False, execution_scope='candidate',
                provenance_limit='Returned runtime binary after launch; no independent device firmware trace')
