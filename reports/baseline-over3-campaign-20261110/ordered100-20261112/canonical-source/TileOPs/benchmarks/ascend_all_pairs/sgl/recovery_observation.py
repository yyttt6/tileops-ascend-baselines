"""Read-only observations of original factory values and returned native calls.

No RNG, alternate input factory, oracle calculation, or candidate replacement.
Tensor copies are taken before a later call could mutate their source storage.
"""
import hashlib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]


class Observation:
    def __init__(self, torch, metadata, destination):
        self.torch, self.metadata, self.destination = torch, metadata, destination
        self.factory_returns, self.calls, self.payloads = [], [], {}

    def snapshot(self, value):
        if isinstance(value, self.torch.Tensor):
            description = self.metadata(self.torch, value)
            cpu = value.detach().cpu().contiguous()
            raw = cpu.reshape(-1).view(self.torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            relative = str((self.destination / (digest + '.bin')).relative_to(WORKSPACE))
            self.payloads[digest + '.bin'] = raw
            description.update(payload_sha256=digest, payload_bytes=len(raw), payload=relative,
                               encoding='contiguous logical tensor bytes; original dtype and shape above')
            return description
        if isinstance(value, (tuple, list)):
            return {'kind':type(value).__name__, 'items':[self.snapshot(v) for v in value]}
        if isinstance(value, dict):
            return {'kind':'dict', 'items':{str(k):self.snapshot(v) for k,v in value.items()}}
        return {'kind':type(value).__name__, 'value':value if isinstance(value,(str,int,float,bool,type(None))) else repr(value)}

    def binary(self, kernel):
        raw = kernel.kernel
        if not isinstance(raw, bytes):
            return {'binary_captured':False, 'binary_type':type(raw).__name__}
        digest = hashlib.sha256(raw).hexdigest()
        name = digest + '.npubin'
        self.payloads[name] = raw
        return {'binary_captured':True, 'binary_sha256':digest, 'binary_bytes':len(raw),
                'binary_payload':str((self.destination/name).relative_to(WORKSPACE)),
                'module_handle':str(kernel.module), 'function_handle':str(kernel.function),
                'binary_origin':'exact returned CompiledKernel.kernel bytes used by unchanged _init_handles/load_binary'}

    def flush(self):
        if not self.payloads:
            return
        self.destination.mkdir(parents=True, exist_ok=False)
        for name, raw in self.payloads.items():
            (self.destination/name).write_bytes(raw)
        self.payloads.clear()
