"""Prepare original-case CPU inputs and reference/control outputs without NPU imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.abc
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / '.cache/all-pairs/sgl/prepared'
SAFE_OPS = set('BatchNormFwdOp ReluFwdOp HardsigmoidFwdOp HardswishFwdOp AdaLayerNormFwdOp AdaLayerNormZeroFwdOp FusedAddLayerNormFwdOp INT8DequantPerTensorFwdOp INT8DequantPerChannelFwdOp INT8DequantPerBlockFwdOp FusedAddRMSNormFwdOp MeanFwdOp InstanceNormFwdOp ArgminFwdOp AminFwdOp VarFwdOp VarMeanFwdOp AbsFwdOp ClampScalarFwdOp ClampTensorFwdOp TanhFwdOp MaskedFillScalarFwdOp MaskedFillTensorFwdOp LerpScalarFwdOp LerpTensorFwdOp WhereFwdOp PreluFwdOp HardtanhFwdOp LeakyReluFwdOp SoftplusFwdOp MulFwdOp MaximumFwdOp MinimumFwdOp SumFwdOp MeanPoolingFwdOp AdaptiveAvgPool2dFwdOp AdaptiveMaxPool2dFwdOp AdaptiveMaxPool2dIndicesFwdOp AvgPool1dFwdOp AvgPool2dFwdOp AvgPool3dFwdOp MaxPool1dFwdOp MaxPool1dIndicesFwdOp MaxPool2dFwdOp MaxPool2dIndicesFwdOp MaxPool3dFwdOp MaxPool3dIndicesFwdOp DeltaNetChunkFwdOp BmmFwdOp Conv1dFwdOp Conv2dFwdOp Conv3dFwdOp GemmFwdOp RMSNormFwdOp NegFwdOp SigmoidFwdOp SiluFwdOp SiluAndMulFwdOp SubFwdOp ArgmaxFwdOp GroupNormFwdOp LayerNormFwdOp'.split())
SAFE_OPS.update('GDNFwdOp KDAFwdOp DeltaNetRecurrentFwdOp FusedQKNormRoPEFwdOp CausalConv1dFwdOp CausalConv1dUpdateFwdOp PagedKVCacheGatherFwdOp MLAKVCacheWriteFwdOp PagedKVCacheWriteFwdOp SamplingFromProbsFwdOp ChainSpeculativeSamplingFwdOp TopKSelectFwdOp TopKMaskFwdOp GroupedGemmFwdOp MoEGroupedGemmFwdOp MoEExpertMLPFwdOp'.split())
SAFE_OPS.update('GQAVarlenFwdOp GQAPagedFwdOp GQAPrefillPagedWithKVCacheFwdOp MHADecodePagedWithKVCacheFwdOp MLADecodeWithKVCacheFwdOp DSADecodeWithKVCacheFwdOp NSAVarlenFwdOp'.split())

SAFE_OPS.update('AnyFwdOp AllFwdOp CountNonzeroFwdOp VectorNormFwdOp'.split())
SAFE_OPS.update('BitwiseAndFwdOp BitwiseOrFwdOp BitwiseXorFwdOp LogicalAndFwdOp LogicalOrFwdOp LogicalNotFwdOp'.split())
SAFE_OPS.update('BmmFP8FwdOp GemmFP8FwdOp GemmINT8W8A8FwdOp'.split())
SAFE_OPS.update('EqFwdOp NeFwdOp GeFwdOp GtFwdOp LeFwdOp LtFwdOp'.split())
SAFE_OPS.add('FP8LightningIndexerFwdOp')
SAFE_OPS.update('ExpFwdOp Expm1FwdOp EluFwdOp SeluFwdOp SignFwdOp'.split())
SAFE_OPS.update('ReciprocalFwdOp DivFwdOp StdFwdOp MishFwdOp'.split())
SAFE_OPS.update('AlibiFwdOp BitwiseNotFwdOp'.split())
SAFE_OPS.update('SoftmaxFwdOp MinPMaskFwdOp'.split())
SAFE_OPS.add('GemmW4A16FwdOp')
SAFE_OPS.add('MoEPostPermuteFwdOp')
SAFE_OPS.add('GeluTanhAndMulFwdOp')
SAFE_OPS.add('DropoutFwdOp')
SAFE_OPS.add('MHCPostFwdOp')
SAFE_OPS.update(('SSDChunkCumsumFwdOp','SSDChunkCouplingFwdOp'))
INPUT_ONLY_OPS = set('DeltaNetFwdOp RoPEFwdOp RoPELlama31FwdOp YaRNFwdOp LongRoPEFwdOp RoPENeoxPositionIdsFwdOp'.split())

def digest(torch, value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str((tuple(t.shape),t.dtype)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v: visit(x)
        elif isinstance(v,dict):
            for k,x in sorted(v.items()):h.update(str(k).encode());visit(x)
        else:h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def move(torch, value, device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,tuple):return tuple(move(torch,x,device) for x in value)
    if isinstance(value,list):return [move(torch,x,device) for x in value]
    if isinstance(value,dict):return {k:move(torch,x,device) for k,x in value.items()}
    return value


def clone(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,tuple):return tuple(clone(torch,x) for x in value)
    if isinstance(value,list):return [clone(torch,x) for x in value]
    if isinstance(value,dict):return {k:clone(torch,x) for k,x in value.items()}
    return value


def main():
    if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
        raise RuntimeError('CPU preparation requires TORCH_DEVICE_BACKEND_AUTOLOAD=0 before Python startup')
    class BlockDevice(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('torch_npu','triton','sgl_kernel_npu','fla'):
                raise ImportError('Device/vendor imports forbidden in CPU preparation: '+fullname)
    sys.meta_path.insert(0,BlockDevice())
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
    for key,sub in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg')]:
        path=ROOT/'.cache/all-pairs/sgl'/sub
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(path)
    import torch
    from benchmarks.api import cases
    from workloads.device import set_run_device
    from tools.baseline_protocol import seed_case
    from workloads.numerics import verify
    torch.set_num_threads(4)
    set_run_device('cpu')
    p = argparse.ArgumentParser()
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--max-mib',type=float,default=512)
    p.add_argument('--report-name',default='preparation')
    p.add_argument('--inputs-only',action='store_true')
    p.add_argument('--min-available-mib',type=float,default=0)
    p.add_argument('--available-multiplier',type=float,default=0)
    args=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    report=[]
    inventory=json.loads((ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/inventory_snapshot_v2.json').read_text())
    bykey={(x['op'],x['case_id']):x for x in inventory['cases']}
    for item in json.loads(args.queue.read_text()):
        op,case_id=item['op'],item['case_id']
        if op not in (INPUT_ONLY_OPS if args.inputs_only else SAFE_OPS):continue
        output=CACHE/f'{op}__{case_id}.pt'
        if output.exists():continue
        spec=bykey[op,case_id]
        import math
        estimate=sum(math.prod(shape)*4 for shape,dtype in spec['tensors'].values())
        if estimate>args.max_mib*1048576:continue
        required=max(args.min_available_mib,estimate/1048576*args.available_multiplier)
        last_notice=0.0
        while required:
            meminfo={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            available=meminfo['MemAvailable']/1024
            if available>=required:break
            if time.monotonic()-last_notice>=30:
                print(json.dumps({'phase':'cpu_memory_wait','op':op,'case_id':case_id,'available_mib':available,'required_mib':required}),flush=True)
                last_notice=time.monotonic()
            time.sleep(5)
        start=time.monotonic()
        record=dict(op=op,case_id=case_id,started_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps({'phase':'cpu_case_start','op':op,'case_id':case_id,'estimated_mib':estimate/1048576}),flush=True)
        try:
            seed=seed_case(torch,op,case_id)
            case=next(c for c in cases(op) if c.id==case_id)
            inputs=case.inputs
            calls={}
            reference=case.reference
            def recorded(*values):
                key=digest(torch,values)
                result=reference(*values)
                calls[key]=clone(torch,result)
                return result
            # Same original numerical evidence exercises its negative controls;
            # no candidate is called and no device result is claimed here.
            if not args.inputs_only:
                verify(recorded,inputs,evidence=case.verification,requests={},preserve_reference_inputs=True)
            sources={}
            for obj in [type(case.workload),reference]:
                path=Path(inspect.getsourcefile(obj)).resolve()
                sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ['TileOPs/workloads/numerics.py','TileOPs/benchmarks/api.py','TileOPs/workloads/workload_base.py','TileOPs/workloads/linear_attention/chunk_cpu.py']:
                sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            payload={'op':op,'case_id':case_id,'seed':seed,'inputs':inputs,'reference_calls':calls,'sources':sources,'input_digest':digest(torch,inputs),'reference_device':None if args.inputs_only else 'cpu','generator':'original manifest workload, run_device=cpu','torch_npu_imported':'torch_npu' in sys.modules,'inputs_only':args.inputs_only}
            temporary=output.with_suffix('.partial')
            torch.save(payload,temporary)
            temporary.rename(output)
            record.update(status='inputs_prepared' if args.inputs_only else 'prepared',reference_calls=len(calls),artifact=str(output.relative_to(ROOT)),artifact_bytes=output.stat().st_size,input_digest=payload['input_digest'],source_sha256=sources)
        except Exception as exc:
            record.update(status='cpu_preparation_failed',error_type=type(exc).__name__,error=str(exc))
        record['seconds']=time.monotonic()-start
        report.append(record)
        with (CACHE/(args.report_name+'.jsonl')).open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('source_sha256','error')}),flush=True)
    print(json.dumps({'phase':'cpu_preparation_finished','cases':len(report),'torch_npu_imported':'torch_npu' in sys.modules}),flush=True)


if __name__=='__main__':main()
