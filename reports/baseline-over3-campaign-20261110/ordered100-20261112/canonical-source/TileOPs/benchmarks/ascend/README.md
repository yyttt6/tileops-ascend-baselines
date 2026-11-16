# Ascend 经典算子 baseline

这里保存 205 个 manifest 算子的分类，以及每类选 1 个代表算子的 baseline 接入结果。当前范围是 13 个代表算子的全部 193 个 case，对照本地 17 个库，共 3281 个 case×库组合。每个组合都有独立记录；不以 example 的固定 shape 或函数名称判定数学语义是否匹配。

## 代码与结果入口

- [算子分类](operator_categories.csv)：205 个算子到类别的映射。
- [范围清单](classic_inventory.json)：分类来源、代表算子、case 参数、17 个库和 manifest 指纹。
- [逐 case 矩阵](classic_matrix.json)：当前结果、原因、设备回执和人工验收证据。
- [按库汇总](classic_library_summary.csv)：每个库的通过数、待测数及范围内不支持数，按通过比例排序。
- [按算子汇总](classic_operator_summary.csv)：至少有一个已验证 baseline 的 case 数，以及通过的 case×库组合数。
- [provider 注册](../ops/bench_ascend_classics.py)：每个算子的具名库入口。
- [共享 resolver](../ascend_baselines.py) 和同目录 `ascend_*.py`：原库加载、归属验证及小型适配器。
- [单 case 检查程序](../../scripts/ascend_classic_check.py)：完整 manifest 输出、输入保护及相关边界检查。

组织方式沿用本仓库 NVIDIA 的共享 baseline resolver 与按算子注册 provider 的模式。case 参数来自统一 workload/manifest，不为每个 shape 复制 kernel 实现。

`pilot_matrix.json`、`pilot_summary.csv` 等 `pilot_*` 文件保留第一阶段的 13×17 单 case 历史快照；全 case 进度以 `classic_*` 文件为准。

## 类别与代表算子

| 类别 | 代表算子 | 本类别算子数 | 代表算子 case 数 |
|---|---|---:|---:|
| Elementwise | AddFwdOp | 70 | 12 |
| RoPE | RoPEFwdOp | 5 | 7 |
| Reduction | AmaxFwdOp | 19 | 5 |
| Normalization | RMSNormFwdOp | 12 | 34 |
| Conv & Pool | Conv2dFwdOp | 16 | 31 |
| GEMM | GemmFwdOp | 8 | 20 |
| Quantization & Dequantization | FP8QuantFwdOp | 10 | 5 |
| Attention | GQADenseFwdOp | 20 | 29 |
| MoE | MoEPrePermuteFwdOp | 12 | 7 |
| Sampling | TopKMaskFwdOp | 6 | 9 |
| Linear Attention | DeltaNetFwdOp | 10 | 18 |
| SSM | SSDChunkStateFwdOp | 10 | 6 |
| Other | FFTC2CFwdOp | 7 | 10 |

## 如何读状态

`verified` 表示该 case 已通过原 manifest 的完整数值检查，并完成原库实现归属验收。它不表示性能胜出，也不保证同一库的其他 case 已通过。当前汇总没有测量性能。

本轮收尾结果为 **870 个已验证组合、0 个待复核组合**。其余 2125 个组合在允许修改范围内不支持，286 个组合保留编译、部署、运行或数值失败等具体原因。这表示限定范围内的逐组合核实已完成，不表示所有组合都运行过或都可用。

三个 packed DeltaNet case 的原 Ascend 参考输出未通过独立 CPU 校验。检查器通过 [参考适配器](../ascend_delta_reference.py) 按逻辑序列分别调用现有 dense 参考，并在每次运行中将完整输出和最终状态与 CPU FP64 逐 token 递推核对，再检查 baseline。序列边界、初始状态、scale、L2 语义和 manifest 容差均保留；原始失败与复验依据记录在矩阵的 `reference_issues` 和设备回执中。

ops-transformer 的 SSD 入口已隔离：6 个 case 均实际运行过，5 个未通过原数值检查；另 1 个 BF16 case 虽通过原绝对容差，补充检查发现原输入的归一化 L2 误差约 1.226，输入乘 8 后有 7.2% 输出超过原容差，因此撤销了其可用记录。历史通过回执仍保留在 `historical_manifest_pass`，新证据见分支 `runs/classics_190/root_review.json`。没有调整容差或修改原生计算来掩盖问题，具体原生故障原因尚未确定。

`pending_case_review` 表示还没有完成当前阶段的核实或设备验收。`unsupported_within_scope` 表示在允许的修改范围内缺少等价路径。`source_*blocked*`、`device_failed` 和 `*_quarantined` 各自保留编译、部署、API、运行或数值失败的证据。超时但没有输出的记录不算数值失败；同入口尚未运行的 case 也不冒充已测试。

适配保留原库的数学计算和 schedule，可做 shape、layout、stride、转置、无损 dtype 表示、padding/cropping 和必要 API 参数映射。不会通过改写 kernel 计算、缓冲策略、同步、融合或数值容差来制造通过结果。所有额外 packing、padding 和 API 调用属于适配器的一部分，未来性能测量需包含其成本。

这里的“原库实现”指回执中指纹对应的本地源码版本，不代表与上游 Git 完全一致。本地库存在编译目标、SDK API、链接、打包等兼容修改；例如 FlashMLA 使用当前 SDK 的公开转换接口，vLLM 内嵌 CATLASS 给已有 VF 方法补充 `static` 声明。最终复核已保存这些 tracked diff、版本和 SHA，见分支证据 `PHASE2/completion_scope_audit_185/audit.json`。没有据此改写 kernel 算法或 schedule。

provider 注册包含开发中与已阻塞的候选入口。复用可用 baseline 时，应先查 `classic_matrix.json` 中对应的 **op、case_id、library** 是否为 `verified`，再依据其回执中的环境和部署信息运行。隔离的 ops-* vendor 与 FLA 原生包需要原测试队列中对应的环境配置，不能仅更改库名套用下面的例子。

## 复查一个已验证 case

以下例子检查 SGL 的 Add case。在工作区根目录运行，整个 Python 进程由主机 NPU 锁包住；输出路径必须尚不存在。

```bash
cd /root/dyq/workspace
npu-guard run --owner 'manual/sgl-add' --wait 600 --timeout 120 -- \
  .tools/baselines/baseline-run sgl-kernel-npu python \
  TileOPs/scripts/ascend_classic_check.py \
  --op AddFwdOp --case-id hidden-state-prefill-float16 \
  --library sgl-kernel-npu \
  --output .cache/ascend-classic/manual/sgl-add-prefill.json
```

该命令生成一次新的检查回执，不会自动更新正式矩阵。正式接纳还要核验源码、部署二进制和原生启动归属。当前上游 pytest Runner 使用 CUDA 计时，这里用 Ascend 检查程序执行设备验证。

仅刷新统计表无需设备：

```bash
python3 TileOPs/scripts/ascend_classic_report.py
```

批量设备测试使用已有主机 `npu-guard` 协议，每批不超过 1 小时，批间连续观察 5 分钟空闲。任务队列、原始回执、验收记录和恢复检查点保存在工作区分支记忆 `.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/`。运行中的候选代码保持冻结，待该批终止并验收后才变更。
