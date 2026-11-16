"""Accepted ops-nn factories; module import never imports any device runtime."""
import json
from pathlib import Path


def _provider(name, accepted_ids, pinned_ids, continuation_ids):
    def factory(case):
        from benchmarks.ascend_baselines import UnsupportedCase
        if case._call.signature.name!=name or case.id not in accepted_ids:
            raise UnsupportedCase('This exact manifest case has no accepted ops-nn numerical and native-ownership receipt; use staging for validation.')
        # Original-only route boundary: historical composed receipts remain intact.
        if name in ('LogicalAndFwdOp', 'LogicalOrFwdOp', 'LogicalNotFwdOp', 'BitwiseNotFwdOp', 'StdFwdOp', 'AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp', 'FusedAddRMSNormFwdOp', 'FusedAddLayerNormFwdOp'):
            raise RuntimeError('Retired composed ops-nn adapter route; historical receipts are preserved. This is not a general original-API absence determination.')
        if name in ('VarFwdOp', 'VarMeanFwdOp') and int(case.params.get('correction', 1)) != 0:
            raise RuntimeError('Retired variance correction composition; original zero-correction auxiliary-output route remains available.')
        if name in ('ClampScalarFwdOp', 'ClampTensorFwdOp', 'HardtanhFwdOp'):
            both_bounds = name == 'HardtanhFwdOp' or (case.params.get('min') is not None and case.params.get('max') is not None)
            if both_bounds:
                raise RuntimeError('Retired two-bound max/min composition; single-bound original API routes remain available.')
        if name in ('CausalConv1dFwdOp', 'CausalConv1dUpdateFwdOp') and case.params.get('activation') == 'silu':
            raise RuntimeError('Retired Conv2DV2 plus Swish composition; original non-SiLU route remains available.')
        if name == 'PagedKVCacheGatherFwdOp' and case.params.get('scale') is not None:
            raise RuntimeError('Retired GatherV2 plus scale multiplication composition; original unscaled route remains available.')
        # Frozen receipts remain evidence; affected MatMulV3 entry is retired.
        if name in {"GroupedGemmFwdOp", "GemmFP8FwdOp", "BmmFP8FwdOp",
                    "MoEGroupedGemmFwdOp", "GemmW4A16FwdOp", "MoEGroupedGemmFP8FwdOp"}:
            import hashlib
            import os
            from benchmarks.ascend_baselines import WORKSPACE
            vendor = Path(os.environ.get("OPS_NN_VENDOR", str(WORKSPACE / ".cache/all-pairs/ops_nn/install3/vendors/allpairs_nn")))
            for proto in vendor.glob("op_proto/lib/linux/*/libcust_opsproto_rt2.0.so"):
                if hashlib.sha256(proto.read_bytes()).hexdigest() == "a7e6ccf94e57413d20c0b4b529c5e10b815331f3dab8e9b4a828f6b06ec5bf7f":
                    raise RuntimeError("MatMulV3 deployment blocked: selected proto contains internal registration modification; deploy pinned original before reuse. This is not an unsupported-case determination.")
        if case.id in continuation_ids:
            import os
            from .continuation_runtime import accepted_records,WS
            record=accepted_records().get((name,case.id))
            if record is None:
                raise UnsupportedCase('Continuation receipt/runtime source hashes no longer match; revalidation is required.')
            if Path(os.environ.get('OPS_NN_VENDOR','')).resolve()!=(WS/record['required_vendor']).resolve():
                raise UnsupportedCase('This exact continuation case requires its recorded install3 vendor before runtime initialization.')
            from .staging import CANDIDATES
            result=CANDIDATES[name](case)
            if result.api!=record['api'] or any(record['source_sha256'].get(k)!=v for k,v in result.source_sha256.items()):
                raise UnsupportedCase('Current candidate differs from its accepted complete receipt.')
            return result
        if case.id in pinned_ids:
            import os
            from dataclasses import replace
            from .preserved_runtime import accepted_records,WS
            if (name,case.id) not in accepted_records():
                raise UnsupportedCase('Preserved runtime source/dependency/receipt hashes no longer match; revalidation is required.')
            required=WS/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'
            if Path(os.environ.get('OPS_NN_VENDOR','')).resolve()!=required:
                raise UnsupportedCase('Preserved smoke004 runtime requires OPS_NN_VENDOR=install3/vendors/allpairs_nn before runtime initialization; see runtime_env.json.')
            from .pinned_smoke004.staging import CANDIDATES as PINNED
            result=PINNED[name](case)
            return replace(result,adaptation='Byte-preserved smoke004 runtime, accepted for this exact case with full output/input/native evidence; relocation introduces no new numerical claim. '+result.adaptation)
        # Historical receipts bind their original factory and classic vendor.
        # A newly validated staging route must not inherit those old case passes.
        from .catalog import LEGACY
        if name in LEGACY:
            from .matrix import legacy
            return legacy(case)
        from .staging import CANDIDATES
        return CANDIDATES[name](case)
    factory.__name__='accepted_'+name
    return factory


_file=Path(__file__).with_name('accepted_cases.json')
_accepted=json.loads(_file.read_text()) if _file.exists() else {}
_pinned_file=Path(__file__).with_name('accepted_pinned_cases.json')
_pinned=json.loads(_pinned_file.read_text()) if _pinned_file.exists() else {}
_continuation_file=Path(__file__).with_name('accepted_continuation_cases.json')
_continuation_data=json.loads(_continuation_file.read_text()) if _continuation_file.exists() else {'records':[]}
_continuation={}
for _record in _continuation_data['records']:
    _continuation.setdefault(_record['op'],set()).add(_record['case_id'])
PROVIDERS={name:_provider(name,frozenset(ids),frozenset(_pinned.get(name,())),frozenset(_continuation.get(name,()))) for name,ids in _accepted.items()}

# Add reviewed exact keys without replacing historical acceptance or fallbacks.
from .verified_runtime import install as _install_verified
_install_verified(PROVIDERS)



# Exact numerical receipt: baseline-integration-700-20261026/nn_math/w4graph01
from .integration700_nn_math_w4graph01 import make as _nn_math_w4graph01
_nn_math_old_w4graph01 = PROVIDERS.get('GemmW4A16FwdOp')
def _nn_math_route_w4graph01(case, _prior=_nn_math_old_w4graph01):
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-l2-resident-float16':
        return _nn_math_w4graph01(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['GemmW4A16FwdOp'] = _nn_math_route_w4graph01

# Exact numerical receipt: baseline-integration-700-20261026/nn_math/adagraph01
from .integration700_nn_math_adagraph01 import make as _nn_math_adagraph01
_nn_math_old_adagraph01 = PROVIDERS.get('AdaLayerNormFwdOp')
def _nn_math_route_adagraph01(case, _prior=_nn_math_old_adagraph01):
    if case._call.signature.name == 'AdaLayerNormFwdOp' and case.id == 'dit-xl-2-float32':
        return _nn_math_adagraph01(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['AdaLayerNormFwdOp'] = _nn_math_route_adagraph01

# Exact numerical receipt: baseline-integration-700-20261026/nn_math/adagraph02
from .integration700_nn_math_adagraph02 import make as _nn_math_adagraph02
_nn_math_old_adagraph02 = PROVIDERS.get('AdaLayerNormFwdOp')
def _nn_math_route_adagraph02(case, _prior=_nn_math_old_adagraph02):
    if case._call.signature.name == 'AdaLayerNormFwdOp' and case.id == 'dit-xl-2-float16':
        return _nn_math_adagraph02(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['AdaLayerNormFwdOp'] = _nn_math_route_adagraph02

# Exact numerical receipt: baseline-integration-700-20261026/nn_math/adagraph03
from .integration700_nn_math_adagraph03 import make as _nn_math_adagraph03
_nn_math_old_adagraph03 = PROVIDERS.get('AdaLayerNormFwdOp')
def _nn_math_route_adagraph03(case, _prior=_nn_math_old_adagraph03):
    if case._call.signature.name == 'AdaLayerNormFwdOp' and case.id == 'dit-xl-2-bfloat16':
        return _nn_math_adagraph03(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['AdaLayerNormFwdOp'] = _nn_math_route_adagraph03

# Exact registration: baseline-integration-700-20261026/nn_registration01
from .integration700_nn_registration01 import make as _nn_registration01_make
_nn_registration01_prior = {}
for _nn_registration01_op in ("FusedAddRMSNormFwdOp", "GemmW4A16FwdOp", "FusedAddLayerNormFwdOp", "GemmFP8FwdOp"):
    _nn_registration01_prior[_nn_registration01_op] = PROVIDERS.get(_nn_registration01_op)
_nn_registration01_keys = {
    ("FusedAddRMSNormFwdOp", "llama-13b-prefill-bfloat16"), ("FusedAddRMSNormFwdOp", "llama-405b-decode-bfloat16"), ("FusedAddRMSNormFwdOp", "llama-405b-prefill-bfloat16"), ("FusedAddRMSNormFwdOp", "llama-70b-decode-bfloat16"), ("FusedAddRMSNormFwdOp", "llama-70b-prefill-bfloat16"), ("FusedAddRMSNormFwdOp", "llama-8b-prefill-bfloat16"), ("FusedAddRMSNormFwdOp", "llama-405b-prefill-float16"), ("FusedAddRMSNormFwdOp", "llama-70b-prefill-float16"), ("FusedAddRMSNormFwdOp", "llama-8b-prefill-float16"),
    ("GemmW4A16FwdOp", "chunked-prefill-tile-float16"), ("GemmW4A16FwdOp", "decode-b32-float16"), ("GemmW4A16FwdOp", "decode-b8-float16"),
    ("FusedAddLayerNormFwdOp", "llama-70b-decode-bfloat16"), ("FusedAddLayerNormFwdOp", "llama-70b-prefill-bfloat16"), ("FusedAddLayerNormFwdOp", "llama-8b-decode-bfloat16"), ("FusedAddLayerNormFwdOp", "llama-8b-prefill-bfloat16"), ("FusedAddLayerNormFwdOp", "llama-13b-prefill-float32"), ("GemmFP8FwdOp", "ds-v3-m1-down-2d-float8_e4m3fn-bfloat16"),
}
def _nn_registration01_route(case, _prior=None):
    _name = case._call.signature.name
    if (_name, case.id) in _nn_registration01_keys:
        return _nn_registration01_make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No accepted exact key")
for _nn_registration01_op, _nn_registration01_old in _nn_registration01_prior.items():
    def _route(case, _prior=_nn_registration01_old):
        return _nn_registration01_route(case, _prior)
    PROVIDERS[_nn_registration01_op] = _route

# Exact registration: baseline700 parent last3 W4
from . import integration700_nn_parent_last3 as _i700_nn_parent_last3
_i700_nn_parent_last3_previous = PROVIDERS["GemmW4A16FwdOp"]
def _i700_nn_parent_last3_route(case):
    if ("GemmW4A16FwdOp", case.id) in _i700_nn_parent_last3._KEYS:
        return _i700_nn_parent_last3.make(case)
    return _i700_nn_parent_last3_previous(case)
PROVIDERS["GemmW4A16FwdOp"] = _i700_nn_parent_last3_route

# Exact registration: baseline-integration-700-20261026/nn_registration02
from importlib.util import spec_from_file_location as _nn_registration02_spec, module_from_spec as _nn_registration02_module
_nn_registration02_s = _nn_registration02_spec('integration700_nn_registration02', '/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_registration02/integration700_nn_registration02.py')
_nn_registration02 = _nn_registration02_module(_nn_registration02_s)
_nn_registration02_s.loader.exec_module(_nn_registration02)
_nn_registration02_prior = PROVIDERS.get('GemmFwdOp')
def _nn_registration02_route(case, _prior=_nn_registration02_prior):
    if (case._call.signature.name, case.id) in _nn_registration02._KEYS:
        return _nn_registration02.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['GemmFwdOp'] = _nn_registration02_route

from . import integration700_nn_remote_remaining6 as _nn_remote_remaining6
for _nn_remote_remaining6_op in ('FusedAddRMSNormFwdOp', 'BatchNormFwdOp'):
    def _nn_remote_remaining6_route(case, _prior=PROVIDERS.get(_nn_remote_remaining6_op)):
        if (case._call.signature.name, case.id) in _nn_remote_remaining6._KEYS:
            return _nn_remote_remaining6.make(case)
        if _prior is not None:
            return _prior(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No registered exact key')
    PROVIDERS[_nn_remote_remaining6_op] = _nn_remote_remaining6_route

# Exact numerical receipts: baseline-integration-700-20261026/nn_dense_resume8
from . import integration700_nn_dense_resume8 as _nn_dense_resume8
_nn_dense_resume8_prior = PROVIDERS.get('GemmFwdOp')
def _nn_dense_resume8_route(case, _prior=_nn_dense_resume8_prior):
    if (case._call.signature.name, case.id) in _nn_dense_resume8._KEYS:
        return _nn_dense_resume8.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact key')
PROVIDERS['GemmFwdOp'] = _nn_dense_resume8_route

# Exact registration: baseline-integration-700-20261026/nn_m2_source_audit1
from importlib.util import spec_from_file_location as _nn_m2_spec, module_from_spec as _nn_m2_module
_nn_m2_s = _nn_m2_spec('integration700_nn_m2_source_audit1', '/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_m2_source_audit1/integration700_nn_m2_source_audit1.py')
_nn_m2 = _nn_m2_module(_nn_m2_s)
_nn_m2_s.loader.exec_module(_nn_m2)
_nn_m2_prior = PROVIDERS.get('GemmFwdOp')
def _nn_m2_route(case, _prior=_nn_m2_prior):
    if (case._call.signature.name, case.id) in _nn_m2._KEYS:
        return _nn_m2.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact GemmFwdOp route')
PROVIDERS['GemmFwdOp'] = _nn_m2_route

# Exact original UnsortedSegmentProd receipt: baseline700/nn_prod_shortpath1.
from . import integration700_nn_prod_shortpath1 as _nn_prod_shortpath1
_nn_prod_shortpath1_prior = PROVIDERS.get('ProdFwdOp')
def _nn_prod_shortpath1_route(case, _prior=_nn_prod_shortpath1_prior):
    if (case._call.signature.name, case.id) == _nn_prod_shortpath1._KEY:
        return _nn_prod_shortpath1.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact Prod route')
PROVIDERS['ProdFwdOp'] = _nn_prod_shortpath1_route

from .integration700_nn_grouped3_build import install as _install_threehost_nn_grouped3_build_700
_install_threehost_nn_grouped3_build_700(PROVIDERS)
