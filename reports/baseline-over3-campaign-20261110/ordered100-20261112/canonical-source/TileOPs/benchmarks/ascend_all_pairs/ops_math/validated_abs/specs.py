"""Inspected native graph API mapping; standard-library-only metadata."""
from pathlib import Path
ROOT = Path('/root/dyq/workspace')
SOURCE = ROOT / 'ascend_baselines/ops-math'
CACHE = ROOT / '.cache/all-pairs/ops_math'
VENDOR = CACHE / 'graph-runtime/vendors/allpairs_math_math'
EVIDENCE = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_math'
# Exact graph names are from each op_host/*def.cpp, not inferred from examples.
SPECS = {}
for name, folder, graph in [
 ('Abs','abs','Abs'),('Neg','neg','Neg'),('Exp','exp','Exp'),('Log','log','Log'),
 ('Sqrt','sqrt','Sqrt'),('Rsqrt','rsqrt','Rsqrt'),('Reciprocal','reciprocal','Reciprocal'),
 ('Sign','sign','Sign'),('Sin','sin','Sin'),('Cos','cos','Cos'),('Floor','floor','Floor'),
 ('Ceil','ceil','Ceil'),('Round','round','Round'),('Trunc','trunc','Trunc'),
 ('Erf','erf','Erf'),('Log1p','log1p','Log1p'),('Expm1','expm1','Expm1'),('Tanh','tanh','Tanh'),
 ('LogicalNot','logical_not','LogicalNot'),('BitwiseNot','bitwise_not','BitwiseNot'),
 ('Isnan','is_nan','IsNan'),('Isinf','is_inf','IsInf'),('Isfinite','is_finite','IsFinite'),
 ('Sub','sub','Sub'),('Mul','mul','Mul'),('Div','div','Div'),('Remainder','floor_mod','FloorMod'),
 ('Pow','pow','Pow'),('FloorDivide','floor_div','FloorDiv'),('Maximum','maximum','Maximum'),
 ('Minimum','minimum','Minimum'),('Eq','equal','Equal'),('Ne','not_equal','NotEqual'),
 ('Gt','greater','Greater'),('Lt','less','Less'),('Ge','greater_equal','GreaterEqual'),
 ('Le','less_equal','LessEqual'),('LogicalAnd','logical_and','LogicalAnd'),
 ('LogicalOr','logical_or','LogicalOr'),('BitwiseAnd','bitwise_and','BitwiseAnd'),
 ('BitwiseOr','bitwise_or','BitwiseOr'),('BitwiseXor','bitwise_xor','BitwiseXor'),
 ('Where','select_v2','SelectV2'),('LerpTensor','lerp','Lerp'),('LerpScalar','lerp','Lerp'),
 ('Amax','reduce_max','ReduceMax'),('Amin','reduce_min','ReduceMin'),
 ('Sum','reduce_sum','ReduceSum'),('Mean','reduce_mean','ReduceMean'),
 ('Prod','reduce_prod','ReduceProd'),('All','reduce_all','ReduceAll'),('Any','reduce_any','ReduceAny'),
 ('LogSumExp','reduce_log_sum_exp','ReduceLogSumExp'),
 ('Argmax','arg_max_v2','ArgMaxV2'),('Argmin','arg_min','ArgMin'),('Cumsum','cumsum','Cumsum'),
 ('NanToNum','nan_to_num','NanToNum'),
]:
 SPECS[name+'FwdOp'] = {'directory':'math/'+folder, 'graph':graph}
for name, folder, graph in [('ClampTensor','clip_by_value','ClipByValue'),('ClampScalar','clip_by_value','ClipByValue'),('Hardtanh','clip_by_value','ClipByValue'),('MaskedFillTensor','masked_fill','MaskedFill'),('MaskedFillScalar','masked_fill','MaskedFill')]:
 SPECS[name+'FwdOp']={'directory':'conversion/'+folder,'graph':graph}
REDUCTIONS={'AmaxFwdOp','AminFwdOp','SumFwdOp','MeanFwdOp','ProdFwdOp','AllFwdOp','AnyFwdOp','LogSumExpFwdOp'}

def files(op):
 root=SOURCE / SPECS[op]['directory']
 return sorted(p for part in ('op_host','op_kernel','op_graph','op_api') for p in (root/part).rglob('*') if p.is_file() and p.suffix in ('.h','.cpp','.cc','.py','.json'))
