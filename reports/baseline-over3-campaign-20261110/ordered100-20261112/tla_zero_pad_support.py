import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import tla_cache_artifact_identity
import tla_zero_pad_candidate
tla_cache_artifact_identity.install()
def candidate_factory(case,library):
 def build():
  candidate=tla_zero_pad_candidate.build(case,library)
  import hashlib
  f=pathlib.Path(tla_cache_artifact_identity.__file__);candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return candidate
 return build
