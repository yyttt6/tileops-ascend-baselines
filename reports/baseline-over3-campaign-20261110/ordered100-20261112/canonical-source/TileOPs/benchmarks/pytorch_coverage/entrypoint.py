"""Prepare an explicit PyTorch baseline using its manifest input factory.

Call ``prepare(case, device='npu')`` inside npu-guard, then invoke
``impl.run(*(impl.args if impl.args is not None else case.inputs))``.
Call ``impl.reset()`` before each repeated invocation when it is present.
The returned object is the standard benchmarks.api.Implementation. This entry
does not select a compiler or claim a performance/correctness result.
"""


def prepare(case, *, device):
    """Generate one input set, apply local factory repairs, and bind the baseline."""
    from workloads.device import set_run_device
    from benchmarks.pytorch_coverage import implementation, make_inputs

    if device not in ('cpu', 'npu'):
        raise ValueError('The explicit baseline supports cpu or npu')
    set_run_device(device)
    # Constructor parameters need no second random input materialization.
    if hasattr(case.workload, 'call'):
        case.workload.arguments = lambda: case._call.arguments({})
    inputs = make_inputs(case, device)
    for name, value in zip(case._call.signature.inputs, inputs, strict=True):
        spec = case._call.specs[name]
        expected_device = 'cpu' if device == 'cpu' or (spec and spec.cpu) else device
        if value is not None and value.device.type != expected_device:
            raise ValueError(f'{name} already has inputs on an incompatible device')
    case.__dict__['inputs'] = inputs
    case.__dict__.pop('verification', None)
    return implementation(case)
