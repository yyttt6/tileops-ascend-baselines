"""Preserve SHA identity for original extension files outside the private workspace."""
import pathlib,hashlib
def hashes(workspace,*paths):
 result={}
 for path in paths:
  path=pathlib.Path(path)
  try:key=str(path.relative_to(workspace))
  except ValueError:key=str(path.absolute())
  result[key]=hashlib.sha256(path.read_bytes()).hexdigest()
 return result
def build(case,library,original_factory):
 from benchmarks.ascend_all_pairs.vllm import runtime
 from benchmarks.ascend_baselines import UnsupportedCase
 if library!='vllm-ascend' or case._call.signature.name!='SoftmaxFwdOp':raise UnsupportedCase('Only original complete Softmax API external-source metadata repair')
 runtime.hashes=lambda *paths:hashes(runtime.WORKSPACE,*paths)
 candidate=original_factory(case,library)()
 candidate.source_sha256.update(hashes(runtime.WORKSPACE,pathlib.Path(__file__)))
 return candidate
