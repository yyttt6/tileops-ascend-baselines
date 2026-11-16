"""Experimental only. Full32 AICPU counted once; stateless single-stream first trial."""
import os,time,json,types,hashlib
from pathlib import Path
from .partition import partition

def make_timer(evidence_dir):
    from tools import measure_baselines as base
    from tools.kernel_task_profile import trace_digest
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    def wrapped(torch, torch_npu, fn, inputs, warmup, repeats, reset=None):
        if reset is not None:raise NotImplementedError("stateful/reset partition not validated")
        pending = []
        session_info = {}
        def one(torch, torch_npu, fn, inputs, reset, flush_kernel, flush_buffer, exclude_names):
            if not pending:
                # Stateless initial trial: use the already-verified 1GiB flush as end marker.
                # No new device kernel/JIT; this differs from CI's 2MiB end marker.
                end_buffer = flush_buffer
                end_marker = flush_kernel
                profile_root = evidence_dir / 'raw'
                profile_root.mkdir()
                old = os.environ.get('ASCEND_WORK_PATH')
                os.environ['ASCEND_WORK_PATH'] = str(profile_root)
                started = time.monotonic()
                try:
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU],
                        schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1,skip_first=0),
                        on_trace_ready=lambda _:None,
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                            l2_cache=False,data_simplification=False)) as prof:
                        for _ in range(repeats):
                            base._isolate_before_call(torch,torch_npu,reset,flush_kernel,flush_buffer)
                            out = base._call_reference(fn, inputs)
                            torch.npu.synchronize()
                            end_marker(end_buffer)
                            torch.npu.synchronize()
                            del out
                        prof.step()
                    session_info['capture_wall_seconds'] = time.monotonic()-started
                    path = Path(prof.prof_if.prof_path)
                    parsed, proof = partition(path,repeats)
                    session_info.update(path=str(path), sha256=trace_digest(path), repeats=repeats)
                    proof['samples']=parsed
                    (evidence_dir/'partition.json').write_text(json.dumps(proof,indent=2))
                    pending.extend(parsed)
                finally:
                    if old is None: os.environ.pop('ASCEND_WORK_PATH',None)
                    else: os.environ['ASCEND_WORK_PATH']=old
            return pending.pop(0)
        timer = types.FunctionType(base._time_reference.__code__,dict(base._time_reference.__globals__,_time_one_isolated_repeat=one),base._time_reference.__name__,base._time_reference.__defaults__)
        result = timer(torch,torch_npu,fn,inputs,warmup,repeats,reset)
        assert not pending
        result['measurement_protocol']='experimental-baseline-full32-marker-session-v1'
        result['single_session']=session_info
        result['profile_coverage_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result
    return wrapped
