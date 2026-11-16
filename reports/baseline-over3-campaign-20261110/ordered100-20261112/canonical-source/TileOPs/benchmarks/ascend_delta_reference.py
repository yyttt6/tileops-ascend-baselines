"""Checked dense-sequence binding for packed DeltaNet benchmark references.

This only adapts the benchmark oracle. No candidate or FLA kernel is changed.
Each complete output is audited against an independent FP64 token recurrence.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys


def packed_delta_reference(case, receipt):
    import numpy as np
    import torch
    from workloads.numerics import compare_outputs

    original_reference = case.reference

    def reference(q, k, v, beta, initial, offsets, offsets_cpu):
        metadata = offsets_cpu if offsets_cpu is not None else offsets
        if metadata is None or q.shape[0] != 1:
            raise ValueError('Packed reference requires explicit sequence boundaries and B=1.')
        boundaries = metadata.cpu().tolist()
        if offsets is not None and offsets_cpu is not None:
            if not torch.equal(offsets.cpu(), offsets_cpu.cpu()):
                raise ValueError('Device and CPU sequence boundaries differ.')
        if (boundaries[0] != 0 or boundaries[-1] != q.shape[1]
                or any(b <= a for a,b in zip(boundaries,boundaries[1:]))):
            raise ValueError('Sequence boundaries must exactly partition the nonempty token axis.')
        originals = [t.detach().cpu().clone() if t is not None else None
                     for t in (q,k,v,beta,initial)]
        qc,kc,vc,bc,hc = originals
        qn,kn,vn,bn = [t.double().numpy() for t in originals[:4]]
        if case.params.get('use_qk_l2norm_in_kernel', False):
            qn = qn / np.sqrt(np.sum(qn*qn,axis=-1,keepdims=True) + 1e-6)
            kn = kn / np.sqrt(np.sum(kn*kn,axis=-1,keepdims=True) + 1e-6)
            qn,kn = [torch.from_numpy(t).to(q.dtype).double().numpy() for t in (qn,kn)]
        scale = case.params.get('scale')
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        output = np.empty(v.shape, dtype=np.float64)
        states = []
        dense_outputs, dense_states = [], []
        for seq,(start,stop) in enumerate(zip(boundaries,boundaries[1:])):
            h = (np.zeros((v.shape[2],q.shape[3],v.shape[3]),dtype=np.float64)
                 if hc is None else hc[seq].double().numpy().copy())
            for token in range(start,stop):
                kt,qt,vt = kn[0,token],qn[0,token],vn[0,token]
                residual = vt - np.einsum('hk,hkv->hv',kt,h)
                h += kt[:,:,None] * (bn[0,token,:,None] * residual)[:,None,:]
                output[0,token] = np.einsum('hk,hkv->hv',qt*scale,h)
            states.append(h)
            dense_inputs = [t[:,start:stop].contiguous() for t in (q,k,v,beta)]
            state = None if initial is None else initial[seq:seq+1].contiguous()
            dense_o,dense_h = original_reference(*dense_inputs,state,None,None)
            dense_outputs.append(dense_o)
            dense_states.append(dense_h)
        expected = (torch.cat(dense_outputs,dim=1),torch.cat(dense_states,dim=0))
        cpu64 = (torch.from_numpy(output).to(q.dtype),torch.from_numpy(np.stack(states)).float())
        dense_cpu = tuple(t.cpu() for t in expected)
        check = compare_outputs(dense_cpu,cpu64,case.verification)
        assert check.checked_outputs == check.total_outputs == 2
        receipt['packed_reference_audit'] = dict(
            passed=True,boundaries=boundaries,verification=asdict(check),
            native_or_tolerance_changes=False,
            method='Original manifest dense reference called independently per logical sequence; both complete outputs checked against CPU FP64 token recurrence with original scale, initial states, native L2 materialization semantics and unchanged manifest tolerances.')
        files = [Path(__file__),Path(__file__).resolve().parents[1]/'workloads/linear_attention/deltanet.py']
        for name,module in tuple(sys.modules.items()):
            path = getattr(module,'__file__',None)
            if name.startswith('fla.') and path and str(path).endswith('.py'):
                files.append(Path(path))
        receipt['reference_source_sha256'] = {
            str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in dict.fromkeys(files)}
        return expected

    return reference
