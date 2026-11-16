"""Layout-only adapters around the original dispatch/combine implementations.

Device runtime imports are inside factories/helpers called by factories. The
CPU audit imports this module without importing torch, torch_npu, ACL or DeepEP.
Only a future externally guarded runtime process may call these factories.
"""

from .contracts import block_owners, paged_indices, plan


def _plan(case, op):
    from benchmarks.ascend_baselines import UnsupportedCase

    try:
        return plan(op, case.params)
    except ValueError as exc:
        raise UnsupportedCase(str(exc)) from exc


def _runtime(case, geometry, op):
    # Loading is deliberately before case.inputs: deployment blockers can be
    # reported without accidentally materializing a device workload.
    from .runtime import open_runtime

    return open_runtime(geometry, {"op": op, "case_id": case.id, "params": case.params})


def _candidate(run, runtime, api, adaptation):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate

    return Candidate(
        Implementation(runtime.observed_run(run)), api, adaptation, runtime.source_hashes,
        limitations=("CPU-audited only; no remote device/numerical verification.",
                     "Owns an isolated rank-0 HCCL process group; call cleanup before another candidate.",
                     "Official JIT successful-load logs plus stable artifact hashes; not per-launch/loaded-byte attestation.",
                     "Provenance-only run: hashing/capture overhead must not be reported as operator latency."),
        timing_mode="provenance_only", cleanup=runtime.close,
    )


def prepermute(case):
    geometry = _plan(case, "MoEPrePermuteFwdOp")
    runtime = _runtime(case, geometry, "MoEPrePermuteFwdOp")
    try:
        import torch

        tokens, topk = geometry["tokens"], geometry["topk"]
        logical_hidden, hidden = geometry["logical_hidden"], geometry["hidden"]

        def run(x, ids):
            # Dispatch uses byte copies for hidden payloads. Reinterpreting FP16
            # as BF16 and back is lossless, even for NaNs; never use .to(BF16).
            payload = x.contiguous().view(torch.bfloat16)
            if logical_hidden != hidden:
                padded = torch.zeros((tokens, hidden), device=x.device, dtype=torch.bfloat16)
                padded[:, :logical_hidden].copy_(payload)
                payload = padded
            rows, _, _, handle, _ = runtime.buffer.dispatch(
                payload, topk_idx=ids.to(torch.int64).contiguous(),
                num_experts=geometry["experts"], expert_alignment=1,
                do_expand=True, do_cpu_sync=True,
            )
            metadata = handle.recv_src_metadata
            if rows.shape != (tokens * topk, hidden) or metadata.shape != (tokens, topk + 2):
                raise RuntimeError("Original dispatch did not return every manifest route.")
            inverse = torch.empty((tokens, topk), device=x.device, dtype=torch.int32)
            inverse[metadata[:, -2].long()] = metadata[:, :topk]
            return (rows[:, :logical_hidden].contiguous().view(x.dtype),
                    handle.psum_num_recv_tokens_per_expert.to(torch.int32), inverse.flatten())

        return _candidate(
            run, runtime, "DeepEP-Ascend::EPBuffer.dispatch(world_size=1, expert_alignment=1)",
            "FP16/BF16 lossless 16-bit payload view; pad/crop hidden width to 128; "
            "convert int32 routing IDs to int64 per call; original native dispatch generates rows, "
            "prefix sums and slots; scatter native slot metadata into inverse route order each run.",
        )
    except BaseException:
        runtime.close()
        raise


def permute_align(case):
    geometry = _plan(case, "MoEPermuteAlignFwdOp")
    runtime = _runtime(case, geometry, "MoEPermuteAlignFwdOp")
    try:
        import torch

        routes, block = geometry["routes"], geometry["block"]

        def run(ids):
            # Flatten routes into separate tokens, so the original dispatch
            # aligns exactly the per-expert route counts, including K>1.
            payload = torch.zeros((routes, geometry["hidden"]), device=ids.device, dtype=torch.bfloat16)
            _, _, _, handle, _ = runtime.buffer.dispatch(
                payload, topk_idx=ids.reshape(routes, 1).to(torch.int64).contiguous(),
                num_experts=geometry["experts"], expert_alignment=block,
                do_expand=True, do_cpu_sync=True,
            )
            metadata = handle.recv_src_metadata
            if metadata.shape != (routes, 3):
                raise RuntimeError("Original dispatch did not return every flattened route.")
            # Native metadata is [destination slot, original route ID, master].
            # This is solely a metadata layout conversion, not a torch sort,
            # histogram, cumsum or replacement permutation implementation.
            sorted_ids = torch.full((geometry["capacity"],), routes, device=ids.device, dtype=torch.int32)
            sorted_ids[metadata[:, 0].long()] = metadata[:, 1]
            owners, padded = block_owners(handle.num_recv_tokens_per_expert_list, block,
                                          geometry["blocks"], geometry["experts"])
            if padded != handle.num_expanded_tokens or padded > geometry["capacity"]:
                raise RuntimeError("Original dispatch padded count disagrees with its output extent.")
            expert_ids = torch.tensor(owners, device=ids.device, dtype=torch.int32)
            count = torch.tensor([padded], device=ids.device, dtype=torch.int32)
            return sorted_ids, expert_ids, count

        return _candidate(
            run, runtime, "DeepEP-Ascend::EPBuffer.dispatch(world_size=1, expert_alignment=block_size)",
            "Flatten [T,K] IDs into [T*K,1] routes each run; original native aligned dispatch "
            "computes all expert counts and destination slots; reformat native source metadata "
            "into sorted route IDs with manifest sentinel padding and padded host counts into block owners; "
            "all synchronization and metadata conversion are timed.",
        )
    except BaseException:
        runtime.close()
        raise


def add(case):
    geometry = _plan(case, "AddFwdOp")
    runtime = _runtime(case, geometry, "AddFwdOp")
    try:
        import torch

        # All-masked routing has no data contribution. The two native epilogue
        # biases perform x+y. This handle depends ONLY on shape and constants;
        # no input data or input-dependent transform is cached at factory time.
        x, _ = case.inputs
        shape, elements = geometry["shape"], geometry["elements"]
        tokens, hidden = geometry["tokens"], geometry["hidden"]
        zeros = torch.zeros((tokens, hidden), device=x.device, dtype=torch.bfloat16)
        masked = torch.full((tokens, 1), -1, device=x.device, dtype=torch.int64)
        empty_rows, _, _, handle, _ = runtime.buffer.dispatch(
            zeros, topk_idx=masked, num_experts=1, expert_alignment=1,
            do_expand=True, do_cpu_sync=True,
        )
        if empty_rows.shape != (0, hidden) or handle.num_recv_tokens != 0:
            raise RuntimeError("All-masked native dispatch unexpectedly retained routes.")

        def pack(value):
            flat = value.expand(shape).contiguous().reshape(-1)
            if flat.numel() != tokens * hidden:
                padded = torch.zeros(tokens * hidden, device=value.device, dtype=value.dtype)
                padded[:elements].copy_(flat)
                flat = padded
            return flat.view(tokens, hidden)

        def run(x, y):
            output, _, _ = runtime.buffer.combine(empty_rows, handle, bias=(pack(x), pack(y)))
            return output.reshape(-1)[:elements].view(shape)

        return _candidate(
            run, runtime, "DeepEP-Ascend::EPBuffer.combine(all-masked handle, bias=(x,y))[0]",
            "BF16 alpha=1; constant all-masked dispatch handle; per-call broadcast/contiguous "
            "reshape to 128-wide rows and optional zero pad/crop; original combine reduce epilogue "
            "adds the two biases with FP32 accumulation and BF16 output; no input-value caching.",
        )
    except BaseException:
        runtime.close()
        raise


def paged_cache_gather(case):
    geometry = _plan(case, "PagedKVCacheGatherFwdOp")
    runtime = _runtime(case, geometry, "PagedKVCacheGatherFwdOp")
    try:
        import torch

        def run(dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
            if scale is not None:
                raise ValueError("DeepEP Engram fetch does not dequantize or scale cache payloads.")
            # Only addresses are translated on the host, never payload values.
            # Every input-dependent transfer/transform is rebuilt inside run.
            indices = paged_indices(
                block_table.cpu().tolist(), cu_seq_lens.cpu().tolist(),
                None if seq_starts is None else seq_starts.cpu().tolist(),
                geometry["page_size"], geometry["pages"], geometry["tokens"],
            )
            addresses = torch.tensor(indices, device=cache.device, dtype=torch.int32).view(1, -1, 1)
            payload = cache.contiguous().view(torch.bfloat16).view(geometry["entries"], geometry["hidden"])
            runtime.buffer.write([payload])
            fetched = runtime.buffer.fetch(addresses)[0]()
            # Preserve dst and cache inputs. Return the complete logical dst;
            # clone also prevents later native buffer reuse changing a receipt.
            return fetched.view(cache.dtype).view(geometry["output_shape"]).clone()

        return _candidate(
            run, runtime, "DeepEP-Ascend::EngramBuffer.write + EngramBuffer.fetch(world_size=1)",
            "FP16/BF16 lossless 16-bit views; flatten cache page rows; translate current "
            "block table/sequence offsets/optional starts into int32 addresses on host each run; "
            "native Engram write stages the full current cache and fetch gathers payloads; "
            "return a private full destination. Host transfers, cache staging and clone are timed.",
        )
    except BaseException:
        runtime.close()
        raise


def count_nonzero(case):
    geometry = _plan(case, "CountNonzeroFwdOp")
    runtime = _runtime(case, geometry, "CountNonzeroFwdOp")
    try:
        import torch

        def run(x):
            rows = x.permute(geometry["permutation"]).contiguous().view(geometry["batches"], geometry["tokens"])
            payload = torch.zeros((geometry["tokens"], geometry["hidden"]), dtype=torch.bfloat16, device=x.device)
            counts = []
            for row in rows:
                # Boolean -> {0,1} int64 is exact. Expert 1's original native
                # histogram is precisely this row's count_nonzero result.
                _, _, _, handle, _ = runtime.buffer.dispatch(
                    payload, topk_idx=row.to(torch.int64).view(-1, 1),
                    num_experts=2, expert_alignment=1, do_expand=True, do_cpu_sync=True,
                )
                counts.append(handle.num_unaligned_recv_tokens_per_expert[1:2].to(torch.int64))
            return torch.cat(counts).view(geometry["output_shape"])

        return _candidate(
            run, runtime, "DeepEP-Ascend::EPBuffer.dispatch(...).handle.num_unaligned_recv_tokens_per_expert[1]",
            "Boolean-only count_nonzero: permute/flatten reduction axes; widen bool to int64 "
            "expert IDs 0/1 exactly; call original native two-expert dispatch once per output row "
            "and select expert 1's count; widen int32 counts to int64 and concatenate. "
            "No Torch predicate, count, sum or histogram; all per-row work is timed.",
        )
    except BaseException:
        runtime.close()
        raise
