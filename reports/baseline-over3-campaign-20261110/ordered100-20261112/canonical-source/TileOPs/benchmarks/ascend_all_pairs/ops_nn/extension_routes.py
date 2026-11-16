"""Independent route additions; leave already tested shared catalog bytes intact."""
ROUTES={'LogSumExpFwdOp':('LogSoftmaxV2','activation/log_softmax_v2')}
APIS={'LogSumExpFwdOp':('AdaptiveMaxPool3d','LogSoftmaxV2','GatherElements','ForeachSubList',
                      'GatherV2','ForeachMulList','ForeachMaximumList','LpNormV2','MaskedScatter')}
