"""FlashMLA providers. Importing this package loads only Python's standard library."""

from .adapters import attention, reduction, pool, mean_pooling, bmm, pointwise_max

PROVIDERS = {
    "GQADenseFwdOp": attention,
    "GQAVarlenFwdOp": attention,
    "GQAPagedFwdOp": attention,
    "GQAPrefillPagedWithKVCacheFwdOp": attention,
    "DSADecodeWithKVCacheFwdOp": attention,
    "NSAVarlenFwdOp": attention,
    "NSACompressedVarlenFwdOp": attention,
    "MLAVarlenFwdOp": attention,
    "AmaxFwdOp": reduction,
    "LogSumExpFwdOp": reduction,
    "MeanFwdOp": reduction,
    "MaxPool1dFwdOp": pool,
    "MaxPool2dFwdOp": pool,
    "MaxPool3dFwdOp": pool,
    "AdaptiveMaxPool2dFwdOp": pool,
    "AvgPool1dFwdOp": pool,
    "AdaptiveAvgPool2dFwdOp": pool,
    "MeanPoolingFwdOp": mean_pooling,
    "BmmFwdOp": bmm,
    "MaximumFwdOp": pointwise_max,
    "ReluFwdOp": pointwise_max,
    "AnyFwdOp": reduction,
}

from .boolean import logical_or
PROVIDERS["LogicalOrFwdOp"] = logical_or
