# DeepEP-Ascend 全量 pair 适配（CPU-only）

入口为 `__init__.py:PROVIDERS`，值均为 `factory(case) -> Candidate`。没有修改共享 registry、旧 classic adapter、vendor、manifest 或 oracle。

任务开始时清单为 205 op / 1982 case；主控在任务中补齐缺失 factory，最新清单为 **205 op / 2062 case**。本目录覆盖最新清单全部行，原始 case 参数原样保留。历史 x86 / CANN 9.2 回执没有计入本次结果。

| 注册 op | 源码合同可适配 case | 原库调用与适配 |
|---|---:|---|
| AddFwdOp | 4 / 12 | BF16、alpha=1；全 masked 常量 dispatch handle，原生 combine 双 bias 完成加法；逐次广播、reshape、必要的 pad/crop |
| CountNonzeroFwdOp | 1 / 11 | 仅 bool；逐输出行无损转 int64 的 0/1 专家编号，原生 dispatch 计数，取专家 1 的计数并无损扩为 int64 |
| MoEPrePermuteFwdOp | 7 / 7 | 原生 expanding dispatch；FP16 以 BF16 位视图搬运后还原，不做数值转换；原生 metadata 转 inverse |
| MoEPermuteAlignFwdOp | 12 / 12 | 将路由展平为 token；原生 expert_alignment=block_size；将原生 slot 与已补齐的 host count 转成三个 manifest 输出 |
| PagedKVCacheGatherFwdOp | 2 / 3 | 非量化 cache；逐次把 block table / offsets / starts 转为地址，原生 Engram write + fetch 搬运数据，返回完整私有输出 |

**26 case 的数学路径为 supported_unverified，当前执行状态为 deployment_blocked；其余 2036 case 为 unsupported。设备验证、数值失败均为 0。** 注册包含具体实现，不包含占位 factory 或 Torch 计算 fallback。

`coverage.json` 包含逐 op / case 状态、具体原因、来源 file/line/SHA256、adapter 路径与回执字段；`coverage_schema.json` 明示字段和状态含义。所有数值、输入保护和原生调用归属回执路径均为 null：CPU 静态检查不能充当设备回执。

重要源码结论：

- combine 的 `topk_weights` 只搬运/恢复，**不乘到 payload**，因此 MoEPostPermute 的加权求和不支持。
- dispatch payload 是字节复制，FP16/BF16 位视图没有丢失精度；不能把此技巧用于原生 BF16 加法。
- dispatch 的专家计数可实现布尔 CountNonzero，但 All/Any 仍需要本库没有的结果谓词，不把有损 count→bool 转换包装成允许的 dtype 适配。
- EngramBuffer 是存储 fetch，不是 Engram gate/convolution；FP8 fetch 也不执行反量化。
- Bucket all_reduce / reduce_scatter 原生代码明确未实现；DeepJIT / clangd 的示例 kernel 和 Torch 测试工具不算 DeepEP 自有实现。

所有依赖输入的转换都在 `run` 中，计入完整 host wall time。仅 Add 的全 masked handle 缓存形状和常量。原输入不被写入；paged gather 返回私有结果。未来设备验收必须覆盖完整 manifest 输出、输入保护、改变输入后重跑、原生调用归属，保留现有 oracle 和容差。

CPU 检查可从 workspace 执行：

```bash
python3 -B -S TileOPs/benchmarks/ascend_all_pairs/deepep/audit.py
python3 -B TileOPs/benchmarks/ascend_all_pairs/deepep/build_host.py
```

第一个检查只用标准库，禁止导入 torch、torch_npu、DeepEP、ACL、TileOPs runtime；验证 205/2062 清单、manifest hash、源码范围/hash、Python AST、惰性注册和 metadata 格式，并检查 26 个 factory 在缺少部署时先报错、不会读取 case.inputs。第二个只用 host C++ 编译器 `-fsyntax-only` 检查 vendor 原始 `csrc/python_api.cpp`，不会加载扩展或运行设备。当前 host 检查明确失败：`third-party/deep_jit/include/deep_jit/utils/exception.hpp:3` 所需的 `elfutils/libdwfl.h` 不存在。诊断及完整命令见 `deployment.json`，原始日志在 `.cache/all-pairs/deepep/host/syntax.log`。

`build_host.py --build --cxx11-abi N` 是后续具备正确编译环境时的纯 host 源码构建入口，ABI 须来自目标 Torch 构建。它不运行 `setup.py`（后者顶层导入 torch_npu），不复制 vendor 的二进制，只在本任务 cache 中构建并生成含架构、Python ABI、全部源码 hash、native hash 的 receipt。`runtime.py` 校验 receipt、Python/header 副本、ELF 架构和真实 API 来源。没有合格 build 时不会导入任何设备 runtime；device/JIT 兼容性仍待未来获准的完整设备验收。

`progress.json` 是可回收状态；任务证据目录的 `cpu-audit-*` 子目录保存增量快照和原始 CPU 编译日志。当前任务没有创建私有 NPU 锁、没有派遣其他 worker，也没有执行任何 NPU 初始化、测试或 profiling。
