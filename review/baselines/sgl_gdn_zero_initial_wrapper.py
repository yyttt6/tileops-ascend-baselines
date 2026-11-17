"""Use the original SGL full API's zero-state mode, without materializing zeros."""
import ast,dataclasses,hashlib,inspect,pathlib,textwrap


def transformed_source(original):
 source=textwrap.dedent(inspect.getsource(original))
 old="""if state is None:
        state=torch.zeros((cu.numel()-1,v.shape[2],q.shape[3],v.shape[3]),device=q.device,dtype=torch.float32)
    elif p.get('state_v_first'):"""
 assert source.count(old)==1,'Original full Triton wrapper changed; no blind rewrite'
 source=source.replace(old,"if state is not None and p.get('state_v_first'):")
 tree=ast.parse(source)
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fn']
 assert len(calls)==1
 kws={x.arg:ast.unparse(x.value) for x in calls[0].keywords}
 assert kws['initial_state']=='state' and kws['output_final_state']=='True' and kws['cu_seqlens']=='cu'
 return source


def _wrap(candidate,run,hashes):
 from benchmarks.ascend_all_pairs.jit_evidence import triton_run
 root=pathlib.Path(inspect.getsourcefile(inspect.unwrap(candidate.implementation.run))).resolve().parents[4]
 wrapped=triton_run(run,hashes,root/'.cache/all-pairs/sgl')
 return dataclasses.replace(candidate,implementation=dataclasses.replace(candidate.implementation,run=wrapped),source_sha256=hashes)


def instrument_original(candidate):
 original=inspect.unwrap(candidate.implementation.run)
 assert original.__module__.endswith('.sgl.linear')
 assert candidate.api=='sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu'
 return _wrap(candidate,original,dict(candidate.source_sha256))


def specialize(case,candidate):
 assert str(case.inputs[0].dtype)=='torch.float16' and case.inputs[5] is None
 assert not any(case.params.get(k,False) for k in ('state_v_first','use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval','use_qk_l2norm_in_kernel'))
 original=inspect.unwrap(candidate.implementation.run)
 assert original.__module__.endswith('.sgl.linear')
 assert candidate.api=='sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu'
 source=transformed_source(original)
 namespace=dict(original.__globals__)
 namespace.update({n:c.cell_contents for n,c in zip(original.__code__.co_freevars,original.__closure__ or (),strict=True)})
 exec(compile(source,str(pathlib.Path(__file__).resolve()),'exec'),namespace)
 path=pathlib.Path(__file__).resolve();hashes={**candidate.source_sha256,str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
 out=_wrap(candidate,namespace[original.__name__],hashes)
 return dataclasses.replace(out,adaptation=candidate.adaptation+' Zero initial state: pass original API initial_state=None; original FP32-zero internal initialization retained; no materialized input state, kernel/precision/sequence boundaries and both outputs unchanged.')
