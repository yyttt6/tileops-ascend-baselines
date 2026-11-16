import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import tla_cache_artifact_identity as identity
identity.install()
from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
def candidate_factory(case,library):
 def build():
  if library!='tilelang-ascend':raise RuntimeError('TLA evidence profile only')
  candidate=original_factory(case,library)()
  import hashlib
  f=pathlib.Path(identity.__file__)
  candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return candidate
 return build
