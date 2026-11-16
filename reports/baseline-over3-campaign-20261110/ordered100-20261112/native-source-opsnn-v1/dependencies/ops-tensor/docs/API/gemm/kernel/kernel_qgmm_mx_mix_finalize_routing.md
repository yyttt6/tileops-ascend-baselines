# KernelQgmmMxMixFinalizeRouting

## 功能说明

`KernelQgmmMxMixFinalizeRouting` 是 `GroupedMatmulFinalizeRouting` 的 MX 全量化 TensorAPI Kernel。组件以分组矩阵乘的 FP32 累加结果为输入，在 AIV 侧完成 logit 乘法、输出类型转换和基于 `rowIndex` 的原子累加写回。

Kernel 使用 `BlockSchedulerGmmSwatWithTailSplit` 在 group 内调度 M/N tile，并通过 AIC/AIV 跨核 flag 约束 MMAD、L0C 到 UB 搬运和向量后处理的执行顺序。

## 模板约束

| 项目 | 约束 |
| --- | --- |
| A/B 数据类型 | 同为 MXFP8（E4M3 或 E5M2）或同为 MXFP4（E2M1 或 E1M2）类型族 |
| WeightNZ | B 布局为 NZ/ZN 时，仅允许 E4M3/E4M3 或 E2M1/E2M1 |
| A 布局 | ND |
| B 布局 | ND、DN、NZ 或 ZN |
| MMAD 输出 | FP32 |
| 最终输出 | FP32 或 BF16 |
| Bias | BF16；由 `hasBias` 指示是否参与 MMAD |

以上为组件层模板约束。算子接口还会基于格式、架构和 dtype 施加更严格的校验。

## 参数

`Params` 包含以下部分：

| 字段 | 说明 |
| --- | --- |
| `problemShape` | 初始 M/N/K 问题规模 |
| `blockMmadParams` | x、weight、scale、bias 的 GM 地址及 MMAD 参数 |
| `prologueParams` | shared input、输出地址和 shared input 缩放参数 |
| `epilogueParams` | y、logit、rowIndex 和 epilogue 参数 |
| `groupListGmAddr` | 分组列表 GM 地址 |
| `gmmParams` | group 数、batch、shared input 范围、基础 tile、L1 分块、scale 分块、bias 和 L0C DB 配置 |

`groupListType` 为累计长度时，Kernel 取相邻元素之差获得当前 group 的 M；为长度列表时直接读取当前元素。

## 执行流程

1. AIV 初始化并执行 Prologue；当存在 shared input 时，将其按 `residualScale` 写入 y 的对应行。
2. AIC/AIV 通过跨核 flag 建立首轮同步，AIV 初始化 Epilogue。
3. Kernel 遍历 groupList，调度当前 group 的 M/N tile；WeightNZ 场景按 NZ 存储规则计算 weight 和 scale 的组偏移。
4. AIC 等待前一 tile 的 AIV 后处理完成，执行 `BlockMmad`。MMAD 结果保持 FP32，并搬运至 Epilogue 提供的 L0C-to-UB 缓冲区。
5. AIC 通知 AIV。AIV 等待该通知后执行 Epilogue：乘 logit、按输出类型完成转换，并按照 `rowIndex` 对 y 做原子加。
6. AIV 通知 AIC 复用下一 tile 的 L0C-to-UB 缓冲；尾部等待确保最后一块后处理完成。

## 依赖组件

| 组件 | 作用 |
| --- | --- |
| `BlockMmadQGmmMx` | MX 量化 MMAD 和 scale 处理 |
| Kernel 内部 Prologue | shared input 预处理 |
| `BlockEpilogueFinalizeRouting` | logit 融合、输出转换和原子写回 |
| `BlockSchedulerGmmSwatWithTailSplit` | group 内 SWAT 调度和尾块切分 |

## 相关文件

- 实现：[kernel_qgmm_mx_mix_finalize_routing.h](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx_mix_finalize_routing.h)
- Epilogue：[block_epilogue_finalize_routing.h](../../epilogue/block/block_epilogue_finalize_routing.md)
