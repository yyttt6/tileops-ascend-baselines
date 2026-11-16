"""Independent oracle/device diagnostics; never promoted to verified coverage."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT.parent)]


def main():
    from check import checker
    checker.require_guard()
    parser=argparse.ArgumentParser()
    parser.add_argument('--op',required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    import torch_npu
    from benchmarks.api import cases
    from benchmarks.ascend_all_pairs.fla import PROVIDERS
    from benchmarks.ascend_baselines import source_hashes
    from tools.baseline_protocol import seed_case
    from workloads.device import set_run_device
    from workloads.numerics import compare_outputs
    set_run_device('npu')
    seed=seed_case(torch,args.op,args.case_id)
    case=next(c for c in cases(args.op) if c.id==args.case_id)
    inputs=case.inputs
    snapshots=[x.clone() if isinstance(x,torch.Tensor) else x for x in inputs]
    candidate=PROVIDERS[args.op](case)
    actual=candidate.implementation.run(*inputs)
    expected_device=case.reference(*inputs)
    row=dict(kind='oracle_diagnostic',op=args.op,case_id=args.case_id,status='diagnostic_complete',
             started_at=datetime.now(timezone.utc).isoformat(),input_seed=seed,
             source_sha256=candidate.source_sha256|source_hashes(Path(__file__),ROOT/'workloads/norm.py'),
             coverage_effect='None: independent diagnostic, original failing numerical receipt is retained.')
    def compare(left,right):
        try:
            result=compare_outputs(left,right,case.verification)
            return dict(passed=True,checked_outputs=result.checked_outputs,total_outputs=result.total_outputs,max_abs_err=result.max_abs_err)
        except Exception as exc:return dict(passed=False,error_type=type(exc).__name__,error=str(exc))
    row['candidate_vs_original_npu_reference']=compare(actual,expected_device)
    def components(left,right):
        left=(left,) if isinstance(left,torch.Tensor) else left
        right=(right,) if isinstance(right,torch.Tensor) else right
        result=[]
        for index,(got,want) in enumerate(zip(left,right,strict=True)):
            if not isinstance(got,torch.Tensor):continue
            a=got.detach().cpu().double();b=want.detach().cpu().double()
            difference=(a-b).abs();position=int(difference.reshape(-1).argmax())
            result.append(dict(component=index,shape=list(got.shape),actual_dtype=str(got.dtype),reference_dtype=str(want.dtype),max_abs_err=float(difference.max()),max_flat_index=position,actual_at_max=float(a.reshape(-1)[position]),reference_at_max=float(b.reshape(-1)[position]),first_actual=a.reshape(-1)[:8].tolist(),first_reference=b.reshape(-1)[:8].tolist()))
        return result
    row['component_diagnostics']=components(actual,expected_device)
    try:
        expected_cpu=checker.cpu_reference(torch,case.reference,inputs)(*inputs)
        row['candidate_vs_original_cpu_reference']=compare(actual,expected_cpu)
        row['original_npu_vs_original_cpu_reference']=compare(expected_device,expected_cpu)
        if args.op=='InstanceNormFwdOp':
            x,rm,rv,weight,bias=inputs
            x=x.cpu().double();rm=rm.cpu().double();rv=rv.cpu().double();spatial=tuple(range(2,x.ndim));shape=(1,x.shape[1],*(1 for _ in spatial));momentum=case.params['momentum'];epsilon=case.params['eps']
            if case.params['use_input_stats']:
                means=x.mean(dim=spatial);variances=x.var(dim=spatial,correction=0)
                high=(x-means.reshape(*x.shape[:2],*(1 for _ in spatial)))*torch.rsqrt(variances.reshape(*x.shape[:2],*(1 for _ in spatial))+epsilon)
                updated_mean=(1-momentum)*rm+momentum*means.mean(dim=0)
                updated_var=(1-momentum)*rv+momentum*x.var(dim=spatial,correction=1).mean(dim=0)
            else:
                high=(x-rm.reshape(shape))*torch.rsqrt(rv.reshape(shape)+epsilon)
                updated_mean,updated_var=rm,rv
            if weight is not None:high=high*weight.cpu().double().reshape(shape)
            if bias is not None:high=high+bias.cpu().double().reshape(shape)
            independent=(high.to(inputs[0].dtype),updated_mean.float(),updated_var.float())
            row['candidate_vs_independent_cpu_fp64_formula_components']=components(actual,independent)
            row['original_reference_vs_independent_cpu_fp64_formula_components']=components(expected_cpu,independent)
            row['diagnostic_note']='Independent FP64 arithmetic exists only in this diagnostic; it is not a candidate, verification oracle replacement, or coverage pass.'
    except Exception as exc:
        row['original_cpu_reference_error']=dict(error_type=type(exc).__name__,error=str(exc))
    for original,after in zip(snapshots,inputs,strict=True):
        if isinstance(original,torch.Tensor):torch.testing.assert_close(original,after,atol=0,rtol=0,equal_nan=True)
    row['inputs_unchanged']=True
    torch.npu.synchronize()
    args.output.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
