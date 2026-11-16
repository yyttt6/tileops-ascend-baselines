"""Reuse the original complete checker; add isolated registry and runtime ownership."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
EVIDENCE = WORKSPACE / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(WORKSPACE)]


def main():
    # Keep compiler temporaries and incidental caches inside this worker's
    # exclusive build root, before loading any framework or vendor package.
    cache=WORKSPACE/'.cache/all-pairs/sgl'
    for key,subdir in (('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('ASCEND_PROCESS_LOG_PATH','ascend-logs')):
        directory=cache/subdir
        directory.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(directory)
    p = argparse.ArgumentParser()
    p.add_argument('--op')
    p.add_argument('--case-id')
    p.add_argument('--queue', type=Path)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('original_classic_checker', ROOT / 'scripts/ascend_classic_check.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.require_guard()
    from benchmarks.ascend_all_pairs.sgl import PROVIDERS
    import benchmarks.ops.bench_ascend_classics as classic
    classic.candidate_factories = lambda case: {'sgl-kernel-npu': lambda: PROVIDERS[case._call.op](case)}
    # Case's operation name is supplied by the immutable queue below (Call API
    # versions differ); only registry selection is substituted, never checking.
    queue = json.loads(args.queue.read_text()) if args.queue else [{'op': args.op, 'case_id': args.case_id}]
    result_dir = EVIDENCE / 'receipts' / args.run
    result_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for item in queue:
        op, case_id = item['op'], item['case_id']
        candidate_calls={'entered':0,'returned':0}
        def build_candidate(case,op=op):
            from dataclasses import replace
            native=PROVIDERS[op](case)
            original=native.implementation.run
            def run(*inputs,**kwargs):
                candidate_calls['entered']+=1
                result=original(*inputs,**kwargs)
                candidate_calls['returned']+=1
                return result
            return replace(native,implementation=replace(native.implementation,run=run))
        classic.candidate_factories = lambda case: {'sgl-kernel-npu': lambda: build_candidate(case)}
        output = result_dir / f'{op}__{case_id}.json'
        if output.exists():
            raise FileExistsError(output)
        called = {}
        native_calls = {}
        from torch.utils._python_dispatch import TorchDispatchMode
        class CaptureNative(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = str(func)
                if name.startswith('allpairs_sgl_unidex.') or name.startswith('npu.') and any(tag in name for tag in ('sgemm', 'bgmv', 'sgmv', 'chunk_kda', 'chunk_gated_delta', 'recurrent_gated_delta', 'causal_conv', 'sparse_flash_attention')):
                    native_calls[name] = native_calls.get(name, 0) + 1
                return func(*args, **(kwargs or {}))
        def trace(frame, event, arg):
            filename = frame.f_code.co_filename
            if event == 'call' and '/ascend_baselines/sgl-kernel-npu/' in filename:
                key = (filename, frame.f_code.co_firstlineno, frame.f_code.co_name)
                called[key] = called.get(key, 0) + 1
        print(json.dumps({'phase':'case_start','op':op,'case_id':case_id}),flush=True)
        sys.argv = ['ascend_classic_check.py', '--op', op, '--case-id', case_id,
                    '--library', 'sgl-kernel-npu', '--output', str(output)]
        sys.setprofile(trace)
        try:
            with CaptureNative():
                current_rc = checker.main()
        finally:
            sys.setprofile(None)
        row = json.loads(output.read_text())
        row['provider_module'] = 'benchmarks.ascend_all_pairs.sgl'
        row['provider_sha256'] = {str(f.relative_to(WORKSPACE)): hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in Path(__file__).parent.glob('*.py')}
        row['runtime_original_calls'] = [{'file': f, 'line': line, 'function': name, 'calls': n}
                                         for (f, line, name), n in sorted(called.items())]
        row['runtime_native_calls'] = native_calls
        row['candidate_calls'] = candidate_calls
        row['attribution_complete'] = candidate_calls['returned']>0 and bool(called or native_calls) and any('/ascend_baselines/sgl-kernel-npu/' in '/' + f for f in row.get('source_sha256', {}))
        if row.get('status') == 'verified' and not row['attribution_complete']:
            row.update(status='attribution_failed', error='No original SGL runtime calls captured')
            current_rc = 1
        output.write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n')
        rc = max(rc, current_rc)
        gc.collect()
        import torch
        torch.npu.synchronize()
        torch.npu.empty_cache()
        if current_rc == 78:
            break
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
