# kernel_wqgmm_mix_weight_prologue

## 功能说明

[kernel_wqgmm_mix_weight_prologue.h](../../../../include/blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h)
提供 arch35 MX A8W4 Grouped Matmul 的通用 Kernel 编排。AIV 将 packed FP4 Weight
转换并搬入共享 L1，AIC 使用 FP8 E4M3 激活和 E8M0 ScaleA/ScaleB 完成 MX MMAD，
并将结果写为 FP16 或 BF16。

Kernel 层拥有 group/block 调度、AIV Prologue、AIC MMAD 调用和同步生命周期；算子侧负责
选择具体数据类型、Layout、BlockMmad、BlockScheduler、BlockPrologue，并把 host tiling
映射到各级 Params。ops-tensor 不提供算子专用 tiling DTO 或一键运行适配函数。

该 Kernel 支持：

- Weight 为 packed FP4 E2M1 或 E1M2；
- 输出和可选 Bias 为 FP16 或 BF16；
- groupListType 为 0（累计 Offset）、1（当前组 Count）或 2（Sparse）；
- 连续多 expert Weight 或每 expert 一个 Tensor 的 TensorList；
- N 方向 main、first-tail、second-tail 三段调度。

## 公共接口

~~~cpp
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"

using KernelImpl = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<
    ProblemShape, BlockMmad, BlockScheduler, void, BlockPrologue,
    IsSingleMultiSingle>;

KernelImpl kernel;
kernel(params);
~~~

模板参数：

| 参数 | 说明 |
| :--- | :--- |
| ProblemShape | 四维 shape，依次保存占位 M、K、N、groupNum |
| BlockMmad | 共享 MX Weight-Prologue Block MMAD 特化 |
| BlockScheduler | Kernel 层 N-resplit 调度器 |
| BlockEpilogue | 当前路径保留为 void |
| BlockPrologue | Kernel 层 FP4→FP8、ScaleB、Bias 前处理 |
| IsSingleMultiSingle | false 表示连续 Weight；true 表示 TensorList |

Kernel Params 包含 ProblemShape、BlockMmad/BlockScheduler/BlockPrologue Params、groupList 地址、
groupListType、hasBias 和 bias 地址。具体 Params 的值必须由算子仓根据自己的 tiling contract 组装。

注意：GroupedMatmul 算子侧 singleWeight 与模板布尔值语义相反。singleWeight=true 使用连续存储并
实例化 IsSingleMultiSingle=false；singleWeight=false 使用 TensorList 并实例化
IsSingleMultiSingle=true。

## 输入、输出与布局

| 参数 | 类型与布局 | 逻辑 shape | 地址语义 |
| :--- | :--- | :--- | :--- |
| x | fp8_e4m3fn_t，ND | [totalM, K] | 所有 expert 沿 M 连续拼接 |
| weight | packed FP4，ZN | 每 expert [K, N] | 连续数据首地址或 TensorList 描述符 |
| xScale | fp8_e8m0_t，ScaleA ND | [totalM, ceil(K/64)*2] | 与 x 的 M 顺序一致 |
| weightScale | fp8_e8m0_t，ScaleB DN | 每 expert [ceil(K/64)*2, N] | 连续数据首地址或 TensorList 描述符 |
| bias | FP16/BF16，ND | 每 expert [N] | hasBias=0 时可为空；否则与 Weight 使用相同存储模式 |
| groupList | int64_t，ND | grouplisttype 0/1：[groupNum]；grouplisttype 2：[groupNum, 2] | 原始数组地址，不是 TensorList |
| y | FP16/BF16，ND | [totalM, N] | 所有 expert 沿 M 连续拼接 |

每个字节承载两个 FP4 元素。连续存储模式下，相邻 expert 的 Weight 字节步长为
K*N/2，ScaleB 元素步长为 N*ceil(K/64)*2，Bias 元素步长为 N。

TensorList 描述符须覆盖 groupNum 项。Weight、ScaleB 和非空 Bias 使用相同的 expert 顺序。

## groupList 与调度约束

groupListType=0 时，第 i 项为前 i+1 个 expert 的累计 M；序列须非递减，最后一项等于
totalM。groupListType=1 时，第 i 项为该 expert 的 M count；所有值之和等于 totalM。
两种模式都允许值为 0 的空 expert。

当 groupListType 为 2 时，groupList 的 shape 为 `[E, 2]`，E 为分组数（groupNum），
数据排布为 `[[groupIdx0, groupSize0], [groupIdx1, groupSize1], ...]`。
其中，groupIdx 表示分组索引，groupSize 表示 M 轴上每组的大小，groupList 中所有数值均须非负。
groupSize 非零的分组按 groupIdx 升序排列在前，groupSize 为零的分组按 groupIdx 升序排列在后。
例如，`[[1, 7], [3, 10], [0, 0], [2, 0]]`。

N 三段 block 必须无遗漏覆盖 N。Kernel 对每个 expert 按 baseM 继续切 M，并在逻辑 Cube core
间轮转起始 block，保持跨 expert 的负载均衡。AIC 和 AIV 使用同一调度器实例语义。

## 调用约束

- global kernel 必须使用 KERNEL_TYPE_MIX_AIC_1_2；
- 当前路径面向 Ascend 950（DAV_3510）；
- 所有 expert 共用 K、N、WeightType 和输出类型；
- hasBias 必须与 bias 地址是否有效一致；
- packed FP4 和 FP8 UB 布局必须按 Align16(N) 建立物理尾块，不能按 N/8 独立分裂；
- BlockMmad 和 BlockPrologue 必须使用同一编译期 grouped schedule policy 和 ready/free 协议。

## 仓内验证

[grouped_matmul_mx_a8w4 example](../../../../examples/grouped_matmul/grouped_matmul_mx_a8w4/README.md)
使用 example-local 适配代码直接实例化通用 Kernel，保留确定性数据生成、NPU 执行和原精度阈值。
Transformer 的生产适配和 host tiling 映射位于 ops-transformer 的
`gmm_weight_quant_tensor_api_mx_kernel.h`。
