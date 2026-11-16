# CATLASS 全算子适配（CPU-only 回执）

当前结果覆盖 **205 个算子、2062 个 manifest case**。任务初始的 1982 是旧 factory case 范围；本目录保留任务期间更新的完整清单及 SHA256。所有状态以 `coverage.json` 的逐 case 记录为准，混合支持算子带 `partial_support` 和分项数量。

| 状态 | 算子数（有该状态的优先聚合） | case 数 |
| --- | ---: | ---: |
| supported_unverified | 17 | 247 |
| deployment_blocked | 26 | 226 |
| unsupported | 162 | 1589 |
| verified / numerical_failed | 0 | 0 |

**没有初始化、导入或使用 NPU，没有加载任何编译后的库，没有设备数值/性能测试，没有复用历史 x86 的 verified 回执。** `static_checks.json` 的 passed 仅表示 CPU 静态检查通过。

## 入口与实现

`__init__.py:PROVIDERS` 是 `dict[str, Callable[[case], Candidate]]`，键为确切 op 名。顶层导入只使用标准库；torch、共享 Candidate helper 和实际动态库加载都在 factory 内。共享 registry、原 manifest、oracle、vendor 源码均未修改。

17 个已注册入口包括：

- GemmFwdOp、BmmFwdOp、GroupedGemmFwdOp、MoEGroupedGemmFwdOp：原 example43 的 FP16/BF16/FP32 类型接口，完整独立 batch/group/expert GEMM；FP32 输出直接选择原生输出类型。
- Conv1dFwdOp、Conv2dFwdOp、Conv3dFwdOp：pad/unfold/stride/transpose 实现完整 im2col，每组调用原始 GEMM 及可选原生 bias，不分解 K reduction。
- GQADenseFwdOp、GQAVarlenFwdOp、GQAPagedFwdOp、MHADecodePagedWithKVCacheFwdOp、GQAPrefillPagedWithKVCacheFwdOp、NSAVarlenFwdOp：原 example49；每次读取当前长度、页表、选择索引，完整 KV 区间调用，保留选择重复项；append 返回完整输出和两份克隆后缓存。
- GemmFP8FwdOp、BmmFP8FwdOp、MoEGroupedGemmFP8FwdOp：原 example51 公共类型参数及已有 FP32 scale epilogue。标量/块 scale 只改布局；逐行 B scale 用独立输出列的完整 GEMM；可选 bias 由原生 epilogue 实现。
- GemmINT8W8A8FwdOp：原 example57 的 INT32 点积和 token/channel FP32 dequantization，BF16 输出。

25 个 DSL 入口已完整实现但未注册：20 个 unary/binary/bitwise/logical/ReLU、Sum、MaxPool1d/2d/3d、AdaptiveMaxPool2d。当前缺失 `ascend_baselines/catlass/python/tla_dsl/catlass/_mlir`，所以 200 个匹配 case 属于部署阻塞。池化只接受完整窗口不超过原始 64 lanes 的 case，没有二次合并部分 reduction。

MLADecodeWithKVCacheFwdOp 另有完整 `mla.py` 和 `native/mla.cpp` 布局桥接，26 个 case 属于部署阻塞：原 example19 数学匹配，但当前3510编译没有 `MmadAtlasA2MLAQK` specialization。修正原始头文件通用 include guard 冲突后，编译仍在原始 dispatch static assertion 失败。没有修改架构标签、schedule 或 vendor 文件，未将此路线注册为可执行。

所有输入变换都在 `run` 内，不缓存输入值或当前 metadata。所有 Candidate 请求 `host_roundtrip` 计时，覆盖 host metadata 读取、布局搬运、多个原生调用和输出恢复的成本。代码中的 torch 仅用于分配、布局/索引、无损表示转换和输出搬运；不调用 torch 的 matmul/卷积/softmax/reduction/激活作为替代实现。

## 可复核证据

- `coverage.json`：schema 说明、205 op/2062 case 结论、理由、file/line/SHA256、适配路径、部署错误、空数值回执字段、manifest 指纹。
- `inventory_snapshot.json`：生成 coverage 时的共享清单原始字节，不改 manifest。
- `source_catalog.json`：原 vendor 代码/文档目录的 1638 个文件指纹及 Python 定义位置；不包括第三方依赖/二进制/历史结果。
- `static_checks.json`：AST/语法、阻断设备相关导入、全清单唯一性/参数不变、473 个兼容 case 的完整输出形状、源码行/哈希、AArch64 ELF 和导出符号检查。
- `build_receipts/`：本次当前源码的 10 种成功编译日志/JSON，及 MLA 兼容编译阻塞日志/JSON。`.so` 仅由本机 CANN9.1 编译到独占 cache，未复制其他机器二进制、未加载。
- `progress.json`：最新进度及未完成设备验证项目；证据 JSON 同步到授权的本任务记忆目录。

仅执行 CPU 静态检查的命令（从 workspace 根目录）：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/catlass/build.py --all
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/catlass/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 TileOPs/benchmarks/ascend_all_pairs/catlass/static_check.py
```

`legacy_build.py` 的失败结果按源码/编译器指纹缓存，避免重复同一失败。审计需要保留本任务 cache 中的 MLA 编译 JSON；回收后的独立主机可使用 `build_receipts/mla-blocked.json` 查阅失败证据，但不得把该编译结果当作新主机回执。

## 剩余限制

数值等价性、完整输出/输入保护和真实调用归属，必须由之后获设备授权的 harness 用原 manifest/oracle 验证。编译成功不能证明运行正确。未改任何容差。大 im2col、逐输出列 FP8 和逐 query/窗口 attention/max pool 路线可能使用大量临时内存或产生大量 launch；未进行性能声明。

不匹配的 case 逐项列出具体契约差异，包括 softcap、RoPE/sink、head width、missing LSE 输出、量化编码、gated/fused/stateful 输出等。`unsupported` 仅针对本源码快照和允许的适配范围，不能解释为永久不可实现。原 FAG 还存在 manifest base2 LSE 与 native natural-log LSE 输入约定差异，记录为接口不匹配，没有修改 oracle 或伪称数值失败。

共享 factory 在任务期间持续补齐；当前缺项及对应 case 在 `coverage.json:missing_workload_factory_ops` / `workload_factory_available` 中记录。这里只核验注册/静态源信息，没有执行 workload 构造。
