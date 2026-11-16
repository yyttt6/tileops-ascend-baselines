# TileLang original-case continuation

This private provider scope does not edit vendor kernels, shared checkers,
registry entries, manifests or references. Results are execution-version-bound;
historical receipts are retained separately from new verification.

`continuation_queue.py` is a CPU-only foreground coordinator. It submits one
synchronous `npu-guard` reservation at a time, using owner
`allpairs/tilelang-continuation`, wait 7200 and hard timeout 300 seconds.
`continuation.py device` stops at a complete-case boundary after the 60-second
soft target. Do not start another coordinator while this owner has a live or
pending reservation. Future cases stay in JSON CPU todo files.

The device phase uses the original case, seed and NPU input generator. It saves
complete input snapshots, original tensor attributes, the original NPU reference
and candidate outputs, source hashes, native observations and synchronization
status. AdaptiveMaxPool2d uses the unchanged reference on exact captured NPU data
in the CPU phase because the original CANN reference has exhibited a device
fault. A synchronization fault ends the device phase without further explicit
device cleanup or large packet serialization.

Only after a terminal guard result exists does the coordinator launch CPU
comparison with `python3 -I -S -B`, Torch backend autoload disabled and device
imports blocked. The unchanged output comparator and input-preservation
comparator process all data. Original device/dtype/shape/layout metadata is
checked separately. The original MoE defined-row validator is retained.
Unimplemented special controls/probes are rejected. A nonzero guard exit cannot
produce a new numerical pass. Small independent CPU packets may be compared by
four processes; large packets remain serial. Exit 75 gets at most one serial
prestart retry, with a fresh tag.

`continuation_trace.py` records successful returns from the original JIT call,
and binds actual returned TVM runtime modules to their binary paths when those
loads are observable. Cold compilation can have a weaker source/cache evidence
layer. Neither compilation, a cached artifact, nor these host observations is
reported as direct driver load/launch proof. No performance result is claimed.

`attention_ir.py` avoids rebuilding static original attention PrimFuncs while
holding the device reservation. The private CPU artifact generator at
`.cache/all-pairs/tilelang_continuation/build_attention_ir.py` constructs the
unchanged original builder for the original parameters, round-trips TVM IR,
and requires byte-identical complete PrimFunc scripts, generated source and
original cache keys against original CPU precompile receipts. The consumer
checks the builder selector AST, kernel/runtime source closure, serialized IR,
restored script, generated source and cache key. Unsealed parameter tuples use
the original builder and compile arguments. Kernel math, precision, tiling and
the FP16 attention prohibition remain unchanged. New attention executions must
be verified with fresh tags after changing this consumer.

Evidence is append-only under
`.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang_continuation`.
Raw complete tensor packets, isolated CPU logs and immutable CPU compilation
artifacts are under `.cache/all-pairs/tilelang_continuation`. `STATE.md` and
the latest checkpoint identify active foreground sessions and CPU todo files.
The original worker event stream, command outputs and historical case console
logs were preserved for independent review; prior nonzero batch exits and
missing native observations remain explicit rather than being promoted to new
guard-zero acceptance.
