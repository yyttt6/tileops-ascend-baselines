"""CPU-only link from observed generated-code hashes to immutable JIT artifacts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang'
CACHE=ROOT/'.cache/all-pairs/tilelang/jit'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
by_hash=defaultdict(list)
for source in CACHE.rglob('device_kernel.asc'):
    binary=source.parent/'executable.so';manifest=source.parent/'manifest.json'
    if not binary.is_file() or not manifest.is_file():continue
    by_hash[digest(source)].append({'generated_source':str(source.relative_to(ROOT)),'executable':str(binary.relative_to(ROOT)),
                                 'executable_sha256':digest(binary),'cache_manifest':str(manifest.relative_to(ROOT)),
                                 'cache_manifest_sha256':digest(manifest)})
observed=[]
for receipt in sorted((OUT/'runs').glob('*.attribution.json')):
    d=json.loads(receipt.read_text())
    for kernel in d['kernels']:
        observed.append({'attribution':str(receipt.relative_to(ROOT)),'kernel_sha256':kernel['kernel_sha256'],
                         'successful_calls':kernel['successful_calls'],'matching_artifacts':by_hash.get(kernel['kernel_sha256'],[])})
result={'schema':'tilelang-compiled-artifacts/v1','method':'Exact SHA256 match of observed JITKernel.get_kernel_source() and isolated-cache device_kernel.asc; same cache directory executable and manifest recorded.',
        'observed_kernel_records':len(observed),'records_with_binary':sum(bool(v['matching_artifacts']) for v in observed),'records':observed}
(OUT/'compiled_artifacts.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
