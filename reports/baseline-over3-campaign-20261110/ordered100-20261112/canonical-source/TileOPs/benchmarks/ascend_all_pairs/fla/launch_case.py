"""Select an original package using CPU metadata, then exec a fresh checker.

Run through baseline-run inside npu-guard. This process imports no torch/vendor
module. exec preserves the reservation's process group and foreground lifetime.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from environment import ROOT,configured_env,mode_for


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--op',required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    inventory=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json'
    row=next(r for r in json.loads(inventory.read_text())['cases'] if (r['op'],r['case_id'])==(args.op,args.case_id))
    mode=mode_for(row['op'],row['tensors'],row['params'])
    env=configured_env(mode)
    evidence=inventory.parent/'fla'
    env['ASCEND_PROCESS_LOG_PATH']=str(evidence/'native_logs'/args.output.stem)
    command=[sys.executable,str(Path(__file__).with_name('check.py')),'--op',args.op,'--case-id',args.case_id,'--library','flash-linear-attention-npu','--output',str(args.output)]
    if args.describe:
        print(json.dumps(dict(mode=mode,command=command,environment={k:env[k] for k in ('FLA_ALLPAIRS_MODE','ASCEND_CLASSIC_FLA_PACKAGE','ASCEND_CUSTOM_OPP_PATH') if k in env}),indent=2))
        return 0
    os.execve(sys.executable,command,env)


if __name__=='__main__':raise SystemExit(main())
