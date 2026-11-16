# DeepGEMM-Ascend: 2026-10-22 continuation

Current evidence: 5 original cases on the current relevant source, 83 passes bound to preserved earlier source versions. New receipts await root review. Coverage spans all 205 operators and 2062 original cases; unsupported rows are not passes. See `resume22_validation.json`, `coverage.json`, and `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/resume22_native/REPORT.md`.

Five new BF16 average-pooling/chunk-mean cases use unchanged original GEMM/Alpha, complete window packing and original divisors. Original NPU factory/RNG inputs and every output are sealed; the original reference and comparison ran on CPU after guard0. The private five-case phase admission patch is not installed in shared scripts. The older 75 root-accepted and eight predecessor extension cases remain version-bound.

Original vendor algorithms, shapes, seeds, factories, oracle and tolerances are unchanged. W4A8 original 12-case oracle holds remain. No performance claim is made. The older narrative below is retained as historical context.

---

# DeepGEMM-Ascend current continuation status — 2026-10-21

The current package has **8 freshly verified original cases** and **75 prior-version passes**. Prior passes retain their exact evidence and are not inherited by changed Python sources. New receipts await root review. `continuation_validation.json`, `progress.json` and `coverage.json` contain the current counts across all 205 operators / 2062 cases for this library.

This update adds original BF16 GEMM Sum/Mean routes through complete axis packing, an exact all-one operand, and native Alpha for Mean. The eight new original cases passed full outputs, input preservation, original native c_return/mapped host ELF, and checker0/child0/outer guard0. New JIT output is isolated by `ALLPAIRS_DEEPGEMM_JIT_CACHE`.

The vendor kernels, original oracle, tolerance, factory, RNG seed, shapes and precision are unchanged. No performance measurement is claimed. W4A8 original 12-case oracle gaps remain held. The old CPU/build narrative below is preserved as historical context; its old zero-pass/deployment statements do not describe current status. Detailed evidence: `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/local_native_continuation/REPORT.md`.

---

# DeepGEMM-Ascend all-pairs CPU 审计与适配

本目录的 `__init__.py:PROVIDERS` 导出 `factory(case) -> Candidate`。入口导入只依赖标准库及本目录模块；不导入 torch、torch_npu、ACL、TileLang 或供应商包。所有设备操作只存在于尚未执行的 factory/run 路径中。

本次已检查 **205 个算子、2062 个 manifest case**。任务开始时清单为 1982 个 case；主控随后补齐 inventory，新增 80 条，本目录按最新快照全部补审。没有修改共享清单、registry、oracle、classic 结果或 vendor 源码。

| Provider | 有具体适配路径的 case | 原库调用 |
|---|---:|---|
| GemmFwdOp | 18 | `bf16_gemm_nt` |
| BmmFwdOp | 9 | `_C.bf16_bmm` |
| GroupedGemmFwdOp | 4 | 每组原生 `bf16_gemm_nt` |
| GemmFP8FwdOp | 8 | 无损 E4M3→BF16，原生 Alpha epilogue |
| BmmFP8FwdOp | 11 | 无损 E4M3→BF16，原生 BMM/Alpha |
| Conv1dFwdOp | 8 | patch 布局与原生 GEMM/bias |
| Conv2dFwdOp | 6 | patch 布局与原生 GEMM/bias |
| Conv3dFwdOp | 2 | patch 布局与原生 GEMM/bias |
| MoEGroupedGemmFwdOp | 8 | 按当前 expert metadata 调用原生 GEMM |
| SSDChunkCouplingFwdOp | 1 | 原生 BMM；逐行 causal crop 与零 padding |

上述 **75 条**的 `semantic_status` 为 `supported_unverified`，`status` 为 `deployment_blocked`。其余 **1987 条**保留具体不支持原因。`verified=0`、设备运行数为 0；没有将历史 x86 上的通过结果提升为本次 ARM64 通过。

适配保留原 kernel 数学和 schedule。卷积支持 stride、padding、dilation、channel groups 和原生 bias，按实际 manifest dtype 筛选；FP16→BF16 有损，未启用。分组输入的 counts/offsets、MoE prefix/masked metadata、FP8 scalar scale 都在每次 `run` 读取，不缓存依赖输入的数据。对应候选使用 `host_roundtrip` 计时标记。patch packing、转置、补零、裁切、输入转换、输出转换及拷贝均在 `run`。

FP8 普通 GEMM/BMM 仅接整张量尺度：E4M3 输入精确表示为 BF16，把两个标量尺度映射到原生 FP32 Alpha epilogue；带 bias 的 GEMM 使用原生 `C=D` 累加模式。尺度乘法的浮点重结合和最终输出仍须原 oracle 验证。任意 FP32 block scale 无法无损表示为 UE8M0，因此没有把重复 block/改变 shape 当作尺度问题的解法。MQA indexer 的原库 BF16 weights/logits 与 manifest FP32 合同不匹配，也没有有损降精度适配。

部署探针仅执行了 CPU 编译器 `-fsyntax-only`：当前 GCC 11.4 缺少 `<format>`；本目录没有可用的本地原生扩展。实际 CANN 为 9.1.0，而供应商 README 要求 9.20；后续 API/JIT 兼容性尚未验证。未执行 `setup.py`，因为它顶层导入 `torch_npu`。`build_host.py` 提供无 torch/vendor 导入的本地 host-extension 构建路线，显式指定匹配 torch 的 C++ ABI 后才能请求构建；本轮仅运行其默认静态探针。

`runtime.py` 只接受本任务 `.cache/all-pairs/deepgemm` 下的构建，检查平台/Python ABI、本地构建回执、二进制及完整 vendor 源码指纹、Python/header 副本和 C++ binding 身份。没有复用或链接其他主机的 `.so`。这些是加载前的归属保护；真实 kernel launch 归属仍需未来设备回执。

产物：

- `coverage.json`：schema `deepgemm-all-pairs-coverage/v1`；逐 op/逐 case 状态、具体原因、源码 file/line/SHA256、adapter，以及空的数值/输入保护/launch 回执字段。`status_definitions` 解释语义支持、部署阻塞、不支持与缺 factory。
- `progress.json`：当前阶段与主控接口问题；证据目录保留初始进度。
- `source_audit.json`：109 个源码指纹、完整 native binding 索引、排除的 ACLNN 参考接口与第三方算子。
- `deployment_probe.json`：确切编译失败和依赖路径；`static_checks.json`：CPU 静态检查回执。
- `audit_coverage.py` / `static_check.py`：标准库重建与审计工具；不实例化 Case、不调用 factory、不分配张量。

共享 factory 仍缺 9 个 op，对应 59 条已枚举 case：`DSAPagedFwdOp`、`FusedAddRMSNormPerTokenQuantFwdOp`、`FusedQKNormRoPEFwdOp`、`GemmW4A8FwdOp`、`HadamardTransformFwdOp`、`MLAKVCacheWriteFwdOp`、`MLAPagedFwdOp`、`MLAVarlenFwdOp`、`SelectiveScanFwdOp`。这些条目的 `workload_factory_available=false`；没有虚构 workload 或测试。

可在 workspace 根目录重新运行 CPU 检查：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python TileOPs/benchmarks/ascend_all_pairs/deepgemm/build_host.py
PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python TileOPs/benchmarks/ascend_all_pairs/deepgemm/audit_coverage.py
PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python TileOPs/benchmarks/ascend_all_pairs/deepgemm/static_check.py
```

详细证据镜像位于 `.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/deepgemm/`，包含当次 inventory 指纹快照。后续仍需兼容工具链、原生构建及另行授权的完整 manifest 输出、输入保护、原库 launch 验证。本次没有发现或修改 oracle 问题。

## 本地 x86 接入进展

root 已使用本地 GCC 13、CANN 9.2.0 beta.2 和 torch C++ ABI 1 完成原始 host
扩展构建，状态为 `host_built_unverified`。报告位于
`.cache/all-pairs/deepgemm/local_deployment_probe.json`，本机构建回执位于
`.cache/all-pairs/deepgemm/package/build_receipt.json`。未修改供应商源码，未复用
远端二进制，也未将编译成功计作设备或数值通过。设备 JIT 与完整 case 验证仍待完成。
`build_host.py --report` 可将新报告写到独立路径，保留上述 ARM64 部署证据。
