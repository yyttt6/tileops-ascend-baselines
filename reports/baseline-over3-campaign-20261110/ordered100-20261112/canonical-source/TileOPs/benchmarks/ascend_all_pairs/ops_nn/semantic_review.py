"""Record source-level conclusions for every manifest operation outside implemented routes."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];WS=ROOT.parent;sys.path.insert(0,str(ROOT))
from benchmarks.ascend_all_pairs.ops_nn.catalog import ROUTES,LEGACY
from benchmarks.ascend_all_pairs.ops_nn.extension_routes import ROUTES as EXTRA_ROUTES
ROUTES=ROUTES|EXTRA_ROUTES
HERE=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn'

def evidence(relative,needle=None):
    p=WS/relative;s=p.read_text();return dict(file=relative,line=next((i for i,t in enumerate(s.splitlines(),1) if needle and needle in t),1),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    inv=json.loads((HERE/'inventory_snapshot.json').read_text());decisions={}
    index=evidence(str((HERE/'source_index.json').relative_to(WS)))
    rules={
      'attention':('The own ops-nn catalogue supplies GEMM, softmax and deformable spatial-attention components, but no equivalent original dense/paged/varlen/MLA/NSA attention or KV-cache state API. Implementing the missing attention/cache algorithm is beyond layout/dtype/API adaptation.', 'ascend_baselines/ops-nn/vfusion/multi_scale_deformable_attn_function/README.md'),
      'linear_attention':('The original recurrent operators implement GRU/LSTM/AUGRU gates. They do not implement the matrix-state delta/GLA/KDA recurrence, chunk state propagation, or its gradients; no layout or exact cast changes that recurrence.', 'ascend_baselines/ops-nn/rnn/CMakeLists.txt'),
      'ssm':('The original recurrent catalogue has GRU/LSTM/AUGRU; it lacks selective/state-space recurrence and the SSD chunk cumsum/coupling/scan/state interfaces. Their state equations are not an equivalent specialization of the available gates.', 'ascend_baselines/ops-nn/rnn/CMakeLists.txt'),
      'rope':('No original own positional-frequency/rotation API in the complete catalogue. Unary sine/cosine are not a RoPE/YaRN/LongRoPE kernel; constructing frequency and pair-rotation arithmetic would implement a new algorithm.', 'ascend_baselines/ops-nn/activation/CMakeLists.txt'),
      'moe':('Original index and GEMM APIs do not expose this routed/fused expert contract including router weights, dispatch/inverse-dispatch metadata and/or expert MLP stages. Independent grouped GEMM has its separate implemented route; synthesizing this remaining routing/fusion algorithm is outside the allowed adaptations.', 'ascend_baselines/ops-nn/index/CMakeLists.txt'),
      'sampling':('The original TopKTopPSample APIs emit sampled indices from logits/sorted candidates; they do not return the manifest dense probability masks or implement the stated probability/speculative RNG contract. Sorting/sampling outputs cannot be recovered as these full outputs by layout adaptation.', 'ascend_baselines/ops-nn/index/top_k_top_p_sample/README.md'),
      'elementwise':('The complete own catalogue has activation and Foreach arithmetic, but not this boolean/bitwise/predicate/generative/remainder/select contract. ForeachBinaryOp op_code is limited to add/sub/mul/div; dtype/layout changes cannot supply the missing computation.', 'ascend_baselines/ops-nn/foreach/foreach_binary_op/op_host/foreach_binary_op_def.cpp'),
      'reduction':('No equivalent own scan/logsumexp original API in the catalogue. Segment reduction APIs do not produce the ordered inclusive prefix sequence, and softmax/logsoftmax discard the normalization constant required by logsumexp.', 'ascend_baselines/ops-nn/index/unsorted_segment_sum/op_host/unsorted_segment_sum_def.cpp'),
      'other':('No own Fourier/Hadamard/MHC/Engram composite implementation in the inspected source catalogue. Existing GEMM, normalization, activation and GRU/LSTM primitives do not by themselves expose the full manifest transform/state/gating contract.', 'ascend_baselines/ops-nn/README.md'),
      'normalization':('No equivalent original fused normalization/position/quantization interface matches all manifest outputs and parameters.', 'ascend_baselines/ops-nn/norm/CMakeLists.txt'),
      'gemm':('Original quantized matrix APIs exist, but this manifest operation has no registered workload factory in the supplied inventory; no numerical test is fabricated.', 'ascend_baselines/ops-nn/matmul/CMakeLists.txt'),
    }
    for op in inv['operators']:
        name=op['op']
        if name in ROUTES or name in LEGACY:continue
        reason,src=rules.get(op['category'],('No equivalent source entry found within the permitted adaptation scope.','ascend_baselines/ops-nn/README.md'))
        if name=='GemmW4A8FwdOp':
            reason='Current factory exists but explicitly has no independent reference: packed-weight order and intermediate dequantization rounding are not defined. Native quantized GEMM APIs exist, but selecting a weight interpretation would invent the oracle contract.'
            src='TileOPs/workloads/ascend_extended_transform_quant.py'
        elif name in ('AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp'):
            reason='Original AdaLayerNorm/AdaLayerNormV2 and Modulate use (1 + scale), whereas this manifest uses scale directly; Zero also requires an extra gate. Shape/stride/dtype/attribute selection alone cannot remove the input-dependent +1 or provide the missing gate.'
            src='ascend_baselines/ops-nn/norm/ada_layer_norm_v2/README.md'
        elif name in ('INT8QuantPerTensorFwdOp','INT8QuantPerChannelFwdOp','INT8QuantPerBlockFwdOp','FP8QuantPerBlockFwdOp','SmoothQuantFwdOp'):
            reason='Manifest dynamic quantization returns scale=1 for an all-zero block. Native DynamicQuant uses max/127 (or reciprocal), and DynamicBlockQuant stores max/FP_MAX including zero. Its min_scale is a reciprocal upper clamp, not the conditional zero->1 rule. Padding/reshape/exact cast cannot repair that input-dependent scale contract; SmoothQuant also has an FP32 smoothing input and exact-code requirements.'
            src='ascend_baselines/ops-nn/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_large_blocksize_kernel.h'
        elif name=='INT4QuantPerGroupFwdOp':
            reason='Manifest requires asymmetric group min/max, exact packed signed nibble codes and both scale/offset, including zero-input control. DynamicQuant supports a symmetric INT4 scale API and DynamicQuantV2 uses a different epsilon/offset convention; the required tuple and zero-group contract are not a layout-only specialization.'
            src='ascend_baselines/ops-nn/quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_base.h'
        elif name=='DropoutFwdOp':
            reason='Own ops-nn activation/control catalogue has no original dropout mask generator/application satisfying this manifest RNG/output contract. No dependency-library or torch dropout fallback is used.'
            src='ascend_baselines/ops-nn/control/CMakeLists.txt'
        elif name in ('IsfiniteFwdOp','IsinfFwdOp','IsnanFwdOp','NanToNumFwdOp'):
            reason='Original ForeachNonFiniteCheckAndUnscale exists and supports ascend950, but it mutates one scalar found_inf flag for the entire TensorList and combines Inf with NaN. It does not expose an elementwise mask, distinguish Inf from NaN, or return the manifest replacement tensor. No equivalent direct original API or implemented scalable representation mapping supplies this contract; this is not a claim that nonfinite detection is absent from the library.'
            src='ascend_baselines/ops-nn/foreach/foreach_non_finite_check_and_unscale/README.md'
        elif name=='FusedAddRMSNormPerTokenQuantFwdOp':
            reason='Own AddRmsNormDynamicQuant/V2 exist, but their normalization multiplies rstd and gamma in FP32 before quantization. The manifest independently rounds normalized values to input dtype before multiplying input-dtype weight, then rounds that product again; it also explicitly caps the returned scale. These input-dependent rounding/cap semantics cannot be supplied by shape/dtype/output-view adaptation of the fused API. This is an API-contract mismatch, not absence of a fused operator.'
            src='ascend_baselines/ops-nn/norm/add_rms_norm_dynamic_quant/op_kernel/arch35/add_rms_norm_dynamic_quant_regbase_common.h'
        elif name.startswith('MaskedFill'):
            reason='Original MaskedScatter can express a constant update stream and is a viable candidate; an isolated package and runtime adapter still require validation. This is not classified as mathematical impossibility.'
            src='ascend_baselines/ops-nn/index/masked_scatter/op_host/masked_scatter_def.cpp'
        elif name=='WhereFwdOp':
            reason='Original Where emits nonzero coordinates, while manifest Where selects elementwise between input/other. Native MaskedScatter consumes a compressed update stream, not the uncompressed input tensor; there is no equivalent direct ternary-select API.'
            src='ascend_baselines/ops-nn/index/where/op_host/where_def.cpp'
        elif name=='FloorDivideFwdOp':
            reason='Original ForeachDivList is true division and has no rounding_mode. Integer floor division, signed correction, and floating floor-division rounding are not an attribute specialization; changing kernel arithmetic is prohibited.'
            src='ascend_baselines/ops-nn/foreach/foreach_div_list/op_host/foreach_div_list_def.cpp'
        if not (WS/src).exists():src='ascend_baselines/ops-nn/README.md'
        manifest=ROOT/op['manifest'] if not op['manifest'].startswith('TileOPs/') else WS/op['manifest']
        ev=[evidence(src),index]
        if name=='FusedAddRMSNormPerTokenQuantFwdOp':ev.append(evidence('TileOPs/workloads/ascend_extended.py','class FusedAddRMSNormQuantCall'))
        if manifest.exists():ev.append(evidence(str(manifest.relative_to(WS)),name+':'))
        decisions[name]=dict(status='oracle_unavailable' if name=='GemmW4A8FwdOp' else 'supported_unverified' if name.startswith('MaskedFill') else 'unsupported_within_scope',reason=reason,sources=ev,scope='Existing own original API equivalence and permitted host layout/shape/stride/padding/exact dtype/API mappings; no native kernel/schedule/oracle/tolerance edits.',device_executed=False)
    (HERE/'semantic_decisions.json').write_text(json.dumps(decisions,indent=2,ensure_ascii=False)+'\n')
    print('Recorded remaining source decisions:',len(decisions))
if __name__=='__main__':main()
