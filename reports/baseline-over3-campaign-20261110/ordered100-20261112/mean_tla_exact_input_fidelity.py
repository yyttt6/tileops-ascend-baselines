from pathlib import Path
import hashlib,json
def verify_runtime_source_identity(P,i,meta):
 P=Path(P);T=P/'workspace/TileOPs';assert i in [1969];current={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for folder in [T/'benchmarks',T/'workloads',T/'src/tileops/manifest'] for f in folder.rglob('*.py')};current[str(P/'reference_policy.py')]=hashlib.sha256((P/'reference_policy.py').read_bytes()).hexdigest();current[str(P/'TASKS.json')]=hashlib.sha256((P/'TASKS.json').read_bytes()).hexdigest();assert current==meta['sources'],'Original source changed since fresh CPU preparation';return True
