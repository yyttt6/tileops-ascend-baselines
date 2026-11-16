"""Source-reviewed API mappings; no runtime imports."""
UNARY = {
    'ReluFwdOp': ('Relu', 'activation/relu'),
    'SigmoidFwdOp': ('Sigmoid', 'activation/sigmoid'),
    'HardsigmoidFwdOp': ('HardSigmoid', 'activation/hard_sigmoid'),
    'HardswishFwdOp': ('HardSwish', 'activation/hard_swish'),
    'LeakyReluFwdOp': ('LeakyRelu', 'activation/leaky_relu'),
    'PreluFwdOp': ('PRelu', 'activation/p_relu'),
    'EluFwdOp': ('Elu', 'activation/elu'),
    'SeluFwdOp': ('Selu', 'activation/selu'),
    'MishFwdOp': ('Mish', 'activation/mish'),
    'SoftplusFwdOp': ('SoftplusV2', 'activation/softplus_v2'),
    'GeluFwdOp': ('GeluV2', 'activation/gelu_v2'),
    'SiluFwdOp': ('Swish', 'activation/swish'),
    'SiluAndMulFwdOp': ('SwiGlu', 'activation/swi_glu'),
    'GeluAndMulFwdOp': ('GeluMul', 'activation/gelu_mul'),
    'GeluTanhAndMulFwdOp': ('GeluMul', 'activation/gelu_mul'),
}
FOREACH = {name+'FwdOp': ('Foreach'+name,'foreach/foreach_'+suffix) for name,suffix in {
    'Abs':'abs','Cos':'cos','Erf':'erf','Exp':'exp','Expm1':'expm1','Log':'log','Log1p':'log1p',
    'Neg':'neg','Reciprocal':'reciprocal','Sign':'sign','Sin':'sin','Sqrt':'sqrt','Tanh':'tanh',
}.items()}
FOREACH.update({name+'FwdOp':('Foreach'+api,'foreach/foreach_'+suffix) for name,api,suffix in [
    ('Div','DivList','div_list'),('Mul','MulList','mul_list'),('Sub','SubList','sub_list'),
    ('Pow','PowList','pow_list'),('Maximum','MaximumList','maximum_list'),('Minimum','MinimumList','minimum_list'),
    ('LerpScalar','LerpScalar','lerp_scalar'),('LerpTensor','LerpList','lerp_list'),
    ('Ceil','RoundOffNumber','round_off_number'),('Floor','RoundOffNumber','round_off_number'),
    ('Round','RoundOffNumber','round_off_number'),('Trunc','RoundOffNumber','round_off_number'),
]})
POOL = {
    'AdaptiveMaxPool2dFwdOp':('AdaptiveMaxPool2d','pooling/adaptive_max_pool2d'),
    'AdaptiveMaxPool2dIndicesFwdOp':('AdaptiveMaxPool2d','pooling/adaptive_max_pool2d'),
    'AdaptiveAvgPool2dFwdOp':('AdaptiveAvgPool2d','pooling/adaptive_avg_pool2d'),
}
for n in (1,2,3):
    POOL[f'AvgPool{n}dFwdOp']=('AvgPool3D','pooling/avg_pool3_d')
    for indices in ('','Indices'):
        POOL[f'MaxPool{n}d{indices}FwdOp']=('MaxPool3DWithArgmaxV2','pooling/max_pool3d_with_argmax_v2')
NORM = {
    'LayerNormFwdOp':('LayerNormV4','norm/layer_norm_v4'),
    'FusedAddLayerNormFwdOp':('AddLayerNorm','norm/add_layer_norm'),
    'FusedAddRMSNormFwdOp':('InplaceAddRmsNorm','norm/inplace_add_rms_norm'),
    'GroupNormFwdOp':('GroupNormV2','norm/group_norm_v2'),
    'InstanceNormFwdOp':('GroupNormV2','norm/group_norm_v2'),
    'BatchNormFwdOp':('BatchNormV3','norm/batch_norm_v3'),
    'BatchNormBwdOp':('BatchNormGradV3','norm/batch_norm_grad_v3'),
    'VectorNormFwdOp':('LpNormV2','norm/lp_norm_v2'),
}
OTHER = {'SoftmaxFwdOp':('SoftmaxV2','activation/softmax_v2'),
         'LogSoftmaxFwdOp':('LogSoftmaxV2','activation/log_softmax_v2'),
         'BmmFwdOp':('BatchMatMulV3','matmul/batch_mat_mul_v3'),
         'Conv1dFwdOp':('Conv2DV2','conv/conv2d_v2'),
         'Conv3dFwdOp':('Conv3DV2','conv/conv3d_v2'),
         'AmaxFwdOp':('AdaptiveMaxPool3d','pooling/adaptive_max_pool3d'),
         'ArgmaxFwdOp':('AdaptiveMaxPool3d','pooling/adaptive_max_pool3d'),
         'MeanFwdOp':('AdaptiveAvgPool2d','pooling/adaptive_avg_pool2d'),
         'INT8DequantPerTensorFwdOp':('ForeachMulList','foreach/foreach_mul_list'),
         'INT8DequantPerChannelFwdOp':('ForeachMulList','foreach/foreach_mul_list'),
         'INT8DequantPerBlockFwdOp':('ForeachMulList','foreach/foreach_mul_list'),
}
LEGACY = {
    'AddFwdOp':('benchmarks.ascend_nn_direct','opsnn_inplace_add','norm/inplace_add_rms_norm'),
    'RMSNormFwdOp':('benchmarks.ascend_rms_cases','opsnn_rms_cases','norm/rms_norm'),
    'GemmFwdOp':('benchmarks.ascend_nn_aligned_gemm','opsnn_aligned_gemm','matmul/mat_mul_v3'),
    'Conv2dFwdOp':('benchmarks.ascend_nn_graph','opsnn_graph_conv','conv/conv2d_v2'),
    'FP8QuantFwdOp':('benchmarks.ascend_quant_cases','opsnn_fp8_cases','quant/dynamic_block_quant'),
}
ROUTES = UNARY | FOREACH | POOL | NORM | OTHER
OTHER.update({
    'CountNonzeroFwdOp':('LpNormV2','norm/lp_norm_v2'),
    'MeanPoolingFwdOp':('AdaptiveAvgPool2d','pooling/adaptive_avg_pool2d'),
    **{name:('ForeachMaximumList + ForeachMinimumList','foreach/foreach_maximum_list') for name in ('ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')},
})
ROUTES.update(OTHER)
ROUTES.update({name:('GatherV2','index/gather_v2') for name in ('IsfiniteFwdOp','IsinfFwdOp','IsnanFwdOp','NanToNumFwdOp')})
ROUTES['BitwiseNotFwdOp']=('ForeachSubList','foreach/foreach_sub_list')
OTHER.update({name:('MatMulV3','matmul/mat_mul_v3') for name in ('GroupedGemmFwdOp','GemmFP8FwdOp','BmmFP8FwdOp')})
ROUTES.update(OTHER)
OTHER.update({name:('LayerNorm','norm/layer_norm') for name in ('VarFwdOp','StdFwdOp','VarMeanFwdOp')})
ROUTES.update(OTHER)
OTHER.update({name:(api,'index/unsorted_segment_'+suffix) for name,api,suffix in [
    ('SumFwdOp','UnsortedSegmentSum','sum'),('ProdFwdOp','UnsortedSegmentProd','prod'),('AminFwdOp','UnsortedSegmentMin','min'),
    ('AllFwdOp','UnsortedSegmentMin','min'),('AnyFwdOp','UnsortedSegmentMax','max')]})
OTHER.update({'ArgminFwdOp':('ForeachNeg + AdaptiveMaxPool3d','foreach/foreach_neg'),'RsqrtFwdOp':('ForeachPowScalar','foreach/foreach_pow_scalar')})
ROUTES.update(OTHER)
OTHER['MoEGroupedGemmFwdOp']=('MatMulV3','matmul/mat_mul_v3')
ROUTES.update(OTHER)
OTHER.update({name:('MatMulV3','matmul/mat_mul_v3') for name in ('GemmW4A16FwdOp','MoEGroupedGemmFP8FwdOp')})
ROUTES.update(OTHER)
OTHER.update({name:('MaskedScatter','index/masked_scatter') for name in ('MaskedFillScalarFwdOp','MaskedFillTensorFwdOp')})
ROUTES.update(OTHER)
OTHER['GemmINT8W8A8FwdOp']=('QuantBatchMatmulV3','matmul/quant_batch_matmul_v3')
ROUTES.update(OTHER)
OTHER.update({name:('LayerNormV4 + ForeachMulList + ForeachSubList','norm/layer_norm_v4') for name in ('AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp')})
ROUTES.update(OTHER)
OTHER.update({name:('Conv2DV2','conv/conv2d_v2') for name in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp')})
ROUTES.update(OTHER)
BOOLEAN_NAMES=('LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp','BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp')
OTHER.update({name:('Original Foreach boolean specialization','foreach/foreach_mul_list') for name in BOOLEAN_NAMES})
ROUTES.update(OTHER)
OTHER['WhereFwdOp']=('GatherElements','index/gather_elements')
ROUTES.update(OTHER)
OTHER['PagedKVCacheGatherFwdOp']=('GatherV2','index/gather_v2')
OTHER.update({name:('ScatterUpdate','index/scatter_update') for name in ('PagedKVCacheWriteFwdOp','MLAKVCacheWriteFwdOp')})
ROUTES.update(OTHER)
ROUTES.update({name:('AdaptiveMaxPool3d','pooling/adaptive_max_pool3d') for name in ('AdaptiveMaxPool2dFwdOp','AdaptiveMaxPool2dIndicesFwdOp')})
