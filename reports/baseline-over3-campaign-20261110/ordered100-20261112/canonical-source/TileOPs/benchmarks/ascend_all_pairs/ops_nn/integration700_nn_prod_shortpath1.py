"""Exact original UnsortedSegmentProd; verified full output and original native cache ELF."""
from pathlib import Path
import hashlib, json, os, types
_KEY = ('ProdFwdOp', 'long-seq-bfloat16')
_P = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_prod_shortpath1')
_VENDOR = '/root/dyq/workspace/.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'
def _load():
    data=(_P/'adapter.py').read_bytes()
    if hashlib.sha256(data).hexdigest()!='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7': raise RuntimeError('Frozen adapter changed')
    m=types.ModuleType('nn_prod_shortpath1_frozen');m.__file__=str(_P/'adapter.py');exec(compile(data,m.__file__,'exec'),m.__dict__);return m
def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name,case.id)!=_KEY: raise UnsupportedCase('No exact receipt')
    if os.environ.get('OPS_NN_VENDOR')!=_VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH')!=_VENDOR: raise UnsupportedCase('Select recorded classicpilot_math before runtime initialization')
    data=(_P/'REGISTER_DEPENDENCIES.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!='b9ec9ce28463bf96630fb3ca02cb6918ed0cb7daa19001c8335ff81f631232e0': raise RuntimeError('Dependency manifest changed')
    for relative,h in json.loads(data).items():
        if hashlib.sha256((Path('/root/dyq/workspace')/relative).read_bytes()).hexdigest()!=h: raise RuntimeError('Dependency changed: '+relative)
    return _load().make(case)
