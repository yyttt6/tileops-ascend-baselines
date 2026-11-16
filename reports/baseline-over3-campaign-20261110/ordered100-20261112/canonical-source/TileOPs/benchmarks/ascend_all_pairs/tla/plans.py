"""Pure metadata capability decisions shared by providers and coverage reporting."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

ELEMENTWISE = 'testing/python/language/test_tilelang_ascend_language_elementwise.py'
UNARY = 'testing/python/base/unary_op.py'
BINARY = 'testing/python/base/binary_op.py'
TILE_API = 'tilelang/language/ascend_tile.py'
FLOATS = {'float16', 'bfloat16', 'float32'}
# All functions are the vendor's original tiled builders, not copied kernels.
DIRECT = {
    'AbsFwdOp': ('vec_abs', 1), 'AddFwdOp': ('vec_add_developer', 2),
    'SubFwdOp': ('vec_sub', 2), 'MulFwdOp': ('vec_mul', 2), 'DivFwdOp': ('vec_div', 2),
    'MaximumFwdOp': ('vec_max', 2), 'MinimumFwdOp': ('vec_min', 2),
    'PowFwdOp': ('vec_pow', 2), 'ExpFwdOp': ('exp', 1), 'LogFwdOp': ('ln', 1),
    'SqrtFwdOp': ('sqrt', 1), 'RsqrtFwdOp': ('rsqrt', 1),
    'ReciprocalFwdOp': ('reciprocal', 1), 'ReluFwdOp': ('relu', 1),
    'SinFwdOp': ('sin', 1), 'CosFwdOp': ('cos', 1),
    'BitwiseAndFwdOp': ('bitwise_and', 2), 'BitwiseOrFwdOp': ('bitwise_or', 2),
    'BitwiseXorFwdOp': ('bitwise_xor', 2), 'BitwiseNotFwdOp': ('bitwise_not', 1),
    'ClampScalarFwdOp': ('clamp', 1), 'HardtanhFwdOp': ('clamp', 1),
    'NegFwdOp': ('vec_muls', 1),
    'LogicalAndFwdOp': ('bitwise_and', 2), 'LogicalOrFwdOp': ('bitwise_or', 2),
    'LogicalNotFwdOp': ('bitwise_xor', 1),
}
GENERIC_UNARY = {'SigmoidFwdOp': 'sigmoid', 'SiluFwdOp': 'silu',
                 'RoundFwdOp': 'round', 'LeakyReluFwdOp': 'leaky_relu'}
COMPARISON = {'EqFwdOp':'EQ','NeFwdOp':'NE','GtFwdOp':'GT','GeFwdOp':'GE',
              'LtFwdOp':'LT','LeFwdOp':'LE','IsnanFwdOp':'NE'}
SELECT = {'WhereFwdOp','MaskedFillScalarFwdOp','MaskedFillTensorFwdOp'}
SPECIAL = {'TanhFwdOp','SiluAndMulFwdOp','GeluTanhAndMulFwdOp','AmaxFwdOp','AminFwdOp',
           'SumFwdOp','SoftmaxFwdOp','GemmFwdOp','BmmFwdOp','RMSNormFwdOp','LayerNormFwdOp',
           'GroupNormFwdOp','MHCPostFwdOp','RoPEFwdOp','RoPENeoxPositionIdsFwdOp',
           'RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','GQADenseFwdOp','GQAVarlenFwdOp',
           'INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp',
           'AllFwdOp','AnyFwdOp','MoEPostPermuteFwdOp','GroupedGemmFwdOp',
           'GemmINT8W8A8FwdOp','CausalConv1dUpdateFwdOp','CausalConv1dFwdOp',
           'InstanceNormFwdOp','MoEGroupedGemmFwdOp','MLADecodeWithKVCacheFwdOp',
           'GQAPagedFwdOp','MHADecodePagedWithKVCacheFwdOp',
           'MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp',
           'Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp','ArgminFwdOp','GQABwdOp'}
IMPLEMENTED = set(DIRECT) | set(GENERIC_UNARY) | set(COMPARISON) | SELECT | SPECIAL


@dataclass(frozen=True)
class Plan:
    supported: bool
    reason: str
    route: str = ''
    dtype: str = ''


def tensor_params(p):
    return {k:v for k,v in p.items() if isinstance(v,(list,tuple)) and len(v)==2
            and isinstance(v[0],(list,tuple)) and isinstance(v[1],str)}


def assess(op, p):
    t = tensor_params(p)
    if op not in IMPLEMENTED:
        return Plan(False, 'No registered equivalent native path; see semantic audit.')
    dt = next((t[k][1] for k in ('input','x','a','q','x_layer_out') if k in t), '')
    if op in DIRECT or op in GENERIC_UNARY or op in COMPARISON or op in SELECT or op in {
        'TanhFwdOp','SiluAndMulFwdOp','GeluTanhAndMulFwdOp'}:
        if op in SELECT:
            dt = t.get('input', t.get('x', ('','')))[1]
        if op.startswith('Logical'):
            if dt != 'bool' and not (op=='LogicalNotFwdOp' and dt in FLOATS):
                return Plan(False,'Boolean bit representation is exact only for bool inputs; no floating truth conversion substitute.')
        elif op.startswith('Bitwise'):
            if dt not in {'bool','int8','uint8','int16','int32','int64'}:
                return Plan(False, f'Original bitwise tiles have no lossless {dt} path.')
        elif dt not in FLOATS and not (dt == 'int32' and op in {
            'AddFwdOp','SubFwdOp','MulFwdOp','MaximumFwdOp','MinimumFwdOp','AbsFwdOp','NegFwdOp'}):
            return Plan(False, f'No lossless native tile specialization for {dt}.')
        if op in {'AddFwdOp','SubFwdOp'} and p.get('alpha',1) != 1:
            return Plan(False, 'Original binary add/sub kernels lack alpha; scaling is not a layout adaptation.')
        if op == 'DivFwdOp' and p.get('rounding_mode') is not None:
            return Plan(False, 'Original division is true division, not floor/trunc division.')
        if op == 'RoundFwdOp' and p.get('decimals',0) != 0:
            return Plan(False, 'Original round tile supports nearest integer only.')
        return Plan(True, 'Original vendor elementwise kernel; complete tiled packing and lossless widening.',
                    'elementwise', dt)
    if op=='GQABwdOp':
        if dt!='float16':
            return Plan(False,'Original GQA backward fixes FP16; BF16 narrowing is forbidden.')
        q=t['q'][0];k=t['k'][0];v=t['v'][0]
        if not p.get('is_causal',False) or q[1]!=k[1] or q[-1]!=128 or v[-1]!=128 or q[1]%64:
            return Plan(False,'Native backward uses equal causal sequences, complete 64-row tiles and 128-wide heads.')
        return Plan(True,'Original native forward/backward pipeline recomputes its own O and natural-log LSE.', 'attention_backward',dt)
    if op=='ArgminFwdOp':
        shape=t['x'][0];dim=p.get('dim');width=prod(shape) if dim is None else shape[dim]
        if dt not in FLOATS or width>2**24:
            return Plan(False,'Native cummin uses FP32 values/indices; no exact integer input or index beyond 2**24.')
        return Plan(True,'Reversed-axis native cumulative min, final index selected and mapped back.', 'argmin',dt)
    if op in {'Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp'}:
        if dt not in {'float16','bfloat16'}:
            return Plan(False,'Native cube GEMM im2col route cannot losslessly narrow FP32 inputs.')
        return Plan(True,'Im2col layout, group slicing and optional homogeneous bias column to original GEMM.', 'convolution',dt)
    if op in {'MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp'}:
        return Plan(True,'Unfold window layout and original native max reduction; no index outputs.', 'max_pool',dt)
    if op=='MLADecodeWithKVCacheFwdOp':
        if dt not in {'float16','bfloat16'}:
            return Plan(False,'Native sparse MLA accepts FP16/BF16 only.')
        heads,dim=t['q'][0][-2:];tail=t['q_pe'][0][-1];length=t['k'][0][1]
        if dim!=512 or tail!=64 or heads not in (64,128) or t['k'][0][2]!=1 or length%64:
            return Plan(False,'Audited complete-key MLA route requires 512+64 features, 64/128 heads, one KV group and complete 64-key tiles; no lossy key padding.')
        return Plan(True,'Original dynamic sparse MLA with every key selected once.', 'mla',dt)
    if op in {'GQAPagedFwdOp','MHADecodePagedWithKVCacheFwdOp'}:
        kp=t.get('k_pages',t.get('k'));vp=t.get('v_pages',t.get('v'))
        if kp[1]!=dt or vp[1]!=dt:
            return Plan(False,'Native paged GQA packing has no dequantization-scale path.')
        inner=dict(p,k=kp,v=vp)
        decision=assess('GQADenseFwdOp',inner)
        return Plan(decision.supported,decision.reason+' Logical KV page gathering is charged per call.', 'paged_attention',dt)
    if op=='MoEGroupedGemmFwdOp':
        if p.get('activation') is not None:
            return Plan(False,'Original expert GEMM has no native fused activation epilogue; vendor MLP uses torch postprocessing.')
        if p.get('out_dtype') not in (None,dt):
            return Plan(False,'Original expert GEMM rounds to input dtype before returning; cannot recover FP32 output.')
        layout=p.get('layout')
        if not isinstance(layout,dict) or not set(layout) & {'masked','contiguous'}:
            return Plan(False,'Layout ADT must be resolved from manifest, not opaque repr.')
        if 'contiguous' in layout and layout['contiguous'].get('metadata_kind')!='physical_psum':
            return Plan(False,'Audited route supports physical prefix sums and masked slabs only.')
        return Plan(True,'Native GEMM per expert; lossless layout slicing with current metadata.', 'moe_gemm',dt)
    if op=='InstanceNormFwdOp':
        if not p.get('use_input_stats',True) or 'running_mean' in t or 'running_var' in t:
            return Plan(False,'Original GroupNorm has no running-statistics input/update contract.')
        return Plan(True,'Original GroupNorm with groups=channels is input-statistics InstanceNorm.', 'instance_group',dt)
    if op=='GroupedGemmFwdOp':
        if dt not in {'float16','bfloat16'}:
            return Plan(False,'Native GEMM template requires FP16/BF16, no lossy narrowing.')
        return Plan(True,'Per-group original GEMM; input-derived offsets read inside each run.', 'grouped',dt)
    if op=='GemmINT8W8A8FwdOp':
        if 'bias' in t:
            return Plan(False,'Original scaled INT8 GEMM has no bias input.')
        return Plan(True,'Original corrected INT8 GEMM with independent row and column scales.', 'int8_gemm',dt)
    if op=='CausalConv1dFwdOp':
        if t['weight'][0][-1]!=4 or p.get('activation')!='silu' or 'bias' in t:
            return Plan(False,'Original prefill is width-4 SiLU without bias.')
        if t['conv_state'][0][0]%2:
            return Plan(False,'Audited prefill route requires pairs of sequences for original two-index cache lookup.')
        return Plan(True,'Original prefill; sequence-pair/channel packing and private FP32 state.', 'causal_prefill',dt)
    if op=='CausalConv1dUpdateFwdOp':
        if t['weight'][0][-1]!=4 or p.get('activation') not in (None,'silu'):
            return Plan(False,'Original decode kernel explicitly loads width 4 and optionally applies SiLU.')
        return Plan(True,'Original causal decode kernel; private transposed history and full state restoration.', 'causal_decode',dt)
    if op.startswith('INT8Dequant'):
        return Plan(True, 'INT8 widens exactly to FP32; broadcast scales to original vendor multiplication kernel.', 'dequant',dt)
    if op == 'MoEPostPermuteFwdOp':
        if p.get('epilogue') is not None:
            return Plan(False, 'Original unpermute lacks routed_scaling_factor epilogue; no external arithmetic substitute.')
        rows=t['expert_output'][0][0]; tokens,topk=t['topk_weights'][0]
        if rows != tokens*topk:
            return Plan(False, 'Original unpermute has a tight T*K row input, not padded/masked expert storage.')
        return Plan(True, 'Original native weighted unpermute; exact FP32 input widening and output cast.', 'moe_post',t['expert_output'][1])
    if op in {'AmaxFwdOp','AminFwdOp','SumFwdOp','SoftmaxFwdOp','AllFwdOp','AnyFwdOp'}:
        if op in {'AllFwdOp','AnyFwdOp'} and dt != 'bool':
            return Plan(False, 'Only boolean input is an exact min/max representation of logical all/any.')
        if dt not in FLOATS and not (dt == 'bool' and op in {'AllFwdOp','AnyFwdOp'}):
            return Plan(False, 'Native reduction kernel lacks exact integer/bool accumulation.')
        if p.get('dtype') not in (None, 'float32', dt):
            return Plan(False, 'Requested accumulator dtype is not a lossless supported representation.')
        if op != 'SoftmaxFwdOp':
            shape=t['x'][0]; axes=p.get('dim',-1)
            axes=list(range(len(shape))) if axes is None else ([axes] if isinstance(axes,int) else axes)
            width=prod(shape[i%len(shape)] for i in axes)
            if width>32768:
                return Plan(False, 'Original whole-row reduction exceeds audited 32768 FP32 element UB bound.')
        return Plan(True, 'Reduction axes moved to a contiguous row; native FP32 reduction; restore shape.',
                    'reduction',dt)
    if op in {'GemmFwdOp','BmmFwdOp'}:
        if dt not in {'float16','bfloat16'}:
            return Plan(False, 'Inspected cube GEMM path uses FP16/BF16; narrowing FP32 is forbidden.')
        return Plan(True, 'Original cube matmul, transpose and zero padding to 128x256x64.', 'gemm',dt)
    if op in {'RMSNormFwdOp','LayerNormFwdOp'}:
        if op == 'RMSNormFwdOp' and 'weight' in t:
            return Plan(False, 'Original RMSNorm builders do not expose affine weight; fused quant path does not return norm output.')
        if op == 'LayerNormFwdOp' and ('weight' in t or 'bias' in t):
            # A group-normalization kernel with groups=1 and S=1 expresses full
            # affine LayerNorm after flattening normalized dimensions to channels.
            return Plan(True, 'Original affine GroupNorm with each row as a batch, G=1, S=1.', 'layer_group',dt)
        return Plan(True, 'Original unweighted normalization with normalized dimensions flattened.', 'norm',dt)
    if op == 'GroupNormFwdOp':
        return Plan(True, 'Original affine group_norm API; preserve groups and epsilon.', 'group_norm',dt)
    if op == 'MHCPostFwdOp':
        return Plan(True, 'Original mhc_post with identity residual mixing and reshape.', 'mhc',dt)
    if op in {'GQADenseFwdOp','GQAVarlenFwdOp'}:
        if dt != 'float16':
            return Plan(False, 'Native GQA kernel fixes dtype=float16; BF16/FP32 narrowing is forbidden.')
        if t['q'][0][-1] != 128 or t['v'][0][-1] != 128:
            return Plan(False, 'Original expert GQA layout is specialized for equal 128-wide QK and V heads.')
        if any(key in t for key in ('sinks','q_scale','k_scale','v_scale')):
            return Plan(False, 'Original GQA lacks quantization-scale and attention-sink inputs.')
        if p.get('softcap') not in (None,0):
            return Plan(False, 'Native GQA has no logit softcap.')
        if p.get('pos_encoding_mode','none') != 'none':
            return Plan(False, 'Native GQA has no fused rotary positional encoding.')
        if p.get('sm_scale') not in (None,128**-.5):
            return Plan(False, 'Native GQA fixes reciprocal-sqrt head scale.')
        return Plan(True, 'Original GQA kernel with logical sequence masks and layout/padding.', 'attention',dt)
    if op in {'RoPEFwdOp','RoPENeoxPositionIdsFwdOp','RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp'}:
        return Plan(True, 'Original rotary kernel; table construction and input clone per invocation.', 'rope',dt)
    return Plan(False, 'Unrecognized route.')
