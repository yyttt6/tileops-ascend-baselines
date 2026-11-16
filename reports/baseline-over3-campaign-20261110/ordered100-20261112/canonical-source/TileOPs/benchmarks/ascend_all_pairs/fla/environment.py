"""CPU-only deployment selection before a fresh runtime process starts."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]


def mode_for(op,tensors,params):
    if op=='SSDStatePassingFwdOp':return 'gdn'
    if op=='SSDChunkCouplingFwdOp':return 'gdn' if tensors['C_mat'][1]=='bfloat16' else 'triton'
    if op in ('SoftplusFwdOp','DeltaNetChunkBwdOp','DeltaNetChunkFwdOp','SSDChunkCumsumFwdOp'):return 'partial_native'
    if op in ('GemmFwdOp','BmmFwdOp','GroupedGemmFwdOp'):
        return 'gdn' if tensors['a'][1]=='bfloat16' else 'triton'
    if op in ('Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp'):
        return 'gdn' if next(iter(tensors.values()))[1]=='bfloat16' else 'triton'
    if op in ('GemmINT8W8A8FwdOp','GemmFP8FwdOp','BmmFP8FwdOp','GLAFwdOp','GLARecurrentFwdOp','GLAChunkFwdOp','ExpFwdOp','Expm1FwdOp'):
        return 'gdn'
    if op in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp'):return 'causal'
    if op in ('DeltaNetRecurrentFwdOp','SigmoidFwdOp','TanhFwdOp'):return 'recurrent'
    if op=='DeltaNetFwdOp':return 'recurrent' if tensors['q'][1]=='bfloat16' and tensors['q'][0][1]<=8 else 'gdn'
    if op=='GDNFwdOp':return 'recurrent' if tensors['q'][1]=='bfloat16' and (params.get('use_gate_in_kernel') or tensors['q'][0][1]<=8) else 'gdn'
    if op=='KDAFwdOp':return 'recurrent' if tensors['q'][1]=='bfloat16' else 'chunk_kda'
    return 'triton'


def configured_env(mode,base=None):
    env=dict(os.environ if base is None else base)
    env['FLA_ALLPAIRS_MODE']=mode
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['TRITON_CACHE_DIR']=str(ROOT/'.cache/all-pairs/fla/triton')
    env['TMPDIR']=str(ROOT/'.cache/all-pairs/fla/tmp')
    env['XDG_CACHE_HOME']=str(ROOT/'.cache/all-pairs/fla/xdg')
    env['TORCH_EXTENSIONS_DIR']=str(ROOT/'.cache/all-pairs/fla/torch_extensions')
    if mode=='recurrent_vfirst':env['FLA_ALLPAIRS_NATIVE_STATE_VFIRST']='1'
    if mode in ('gdn','gdn_matmul','recurrent','recurrent_vfirst','chunk_kda','new_native','partial_native'):
        if mode=='partial_native':
            package=ROOT/'.cache/all-pairs/fla/python_binary_partial/fla_npu'
        elif mode in ('chunk_kda','new_native'):
            ready=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla/kda_native_ready.json'
            package=ROOT/('.cache/all-pairs/fla/python_binary_kda/fla_npu' if mode=='chunk_kda' and ready.is_file() else '.cache/all-pairs/fla/python_binary/fla_npu')
        else:
            suffix='flash-linear-attention-npu'+('-recurrent-kda' if mode in ('recurrent','recurrent_vfirst') else '')
            package=ROOT/'.cache/ascend-classic/native-pilot-20261020'/suffix/'python_binary/fla_npu'
        env['ASCEND_CLASSIC_FLA_PACKAGE']=str(package)
        env['PYTHONPATH']=str(package.parent)+os.pathsep+env.get('PYTHONPATH','')
        vendor=package/'opp/vendors/fla_npu_transformer'
        env['ASCEND_CUSTOM_OPP_PATH']=str(vendor)+os.pathsep+str(package/'opp')
        env['LD_LIBRARY_PATH']=str(vendor/'op_api/lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    return env
