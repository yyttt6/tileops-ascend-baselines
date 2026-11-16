# Kernel Qgmm Mx
> 基础 QGMM [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx.h)；
> ActivationQuant [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h)

## 功能说明
MX 量化 Grouped Matmul 的 Kernel 组件，基于 Tensor API 实现。普通 QGMM 与 GMMAQ
ActivationQuant 分别由 `kernel_qgmm_mx.h` 和 `kernel_qgmm_mx_activation_quant.h`
独立实现，不通过共享 Kernel 基类耦合。普通 QGMM 调度标签仅支持 AIC；
`KernelGroupedMmadWithScaleMxActivationQuant` 启用 AIC:AIV=1:2 的融合后处理路径。
普通 QGMM Kernel 自己完成 group list 解析、Group 偏移和 Tensor 组装；GMMAQ 路径额外使用
Scheduler 的 MX group/block 偏移接口完成 AIC/AIV 协同。

**实现框架参考**：[Kernel 公共框架](./kernel.md)

## 特殊约束

### 计算模式
默认标签仅支持 AIC。ActivationQuant 标签由 AIC 将 float L0C 写入 UB，两个 AIV 执行
GeluTanh 与 MXFP8/MXFP4 在线量化，并通过跨核 flag 串行复用 UB。

### Block 依赖
通常与 `BlockMmad<GroupedMatmulWithScaleMx<...>, ...>` 搭配使用。

### 调度器限制
使用 `BlockSchedulerGmmSwatWithTailSplit` 负责 grouped matmul 的单核 block 分发和 tail split。
ActivationQuant 标签保持原融合核语义，不对最后一组启用 tail split，避免 64 元素 MX scale
组被更小 N 子块重叠写入。

### Scale 类型
ScaleA 和 ScaleB 固定按 `fp8_e8m0_t` 解释。

### Dtype / Format 静态校验
Kernel 编译期通过 `static_assert` 校验模板组合：

- `AType` / `BType` 仅支持同 bit-width 的 MXFP8 组合（`fp8_e4m3fn_t`、`fp8_e5m2_t`）或 MXFP4 组合（`fp4x2_e2m1_t`、`fp4x2_e1m2_t`）。
- `CType` 支持 `half`、`bfloat16_t`、`float`；`BiasType` 仅支持 `float`。
- `LayoutA` 仅支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`。
- `LayoutB` 仅支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- `LayoutC` / `LayoutBias` 支持 ND 类布局，不支持 `nz_layout_ptn`、`zn_layout_ptn`。

### SwiGLU MX 融合路径约束
`KernelGmmSwiGluMixMx` 路径复用 QGMM MX 的 Cube 计算并在 AIV 侧完成 SwiGLU 与 MX 输出量化。当前该路径仅支持 MXFP8 输入：

- `AType` / `BType` 仅支持 MXFP8（`fp8_e4m3fn_t`、`fp8_e5m2_t`）。
- `CType` / `BiasType` 仅支持 `float`。
- `LayoutA` 仅支持 `nd_ext_layout_ptn`。
- `LayoutB` 支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- `LayoutC` / `LayoutBias` 支持 ND 类布局，不支持 `nz_layout_ptn`、`zn_layout_ptn`。

完整的双 N 半区、NZ/ZN 双源搬运、shape、mode 和空指针契约见
[Kernel QGMM SwiGLU MX](./kernel_qgmm_swiglu_mx.md)。

## 特殊类型别名

| 别名 | 含义 |
|------|------|
| `ProblemShape` | 整体问题规模 |
| `SchedulerProblemShape` | scheduler 使用的当前 group 问题规模 |
| `BlockShape` | 单核 block 形状 |
| `SchedulerBlockShape` | scheduler 返回的单核基本块大小与 tail split 偏移 |
| `BlockCoord` | block 坐标 |

## 特殊数据结构

### GMMTiling
```cpp
struct GMMTiling {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t scaleKAL1;
    uint32_t scaleKBL1;
    uint8_t isBias;
    uint8_t dbL0C;
    uint8_t l1BufferStage;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
};
```

参数说明：

| 参数 | 说明 |
|------|------|
| `groupNum` | group 数量 |
| `m/n/k` | 初始问题规模 |
| `baseM/baseN/baseK` | 基础 block/L0 tile 大小 |
| `kAL1/kBL1` | A/B 的 L1 K 轴切分 |
| `scaleKAL1/scaleKBL1` | ScaleA/ScaleB 的 L1 K 轴切分 tiling 字段；在 MX 量化中两者必须一致，并作为 BlockMmad 的共享 `scaleKL1` |
| `isBias` | 是否启用 bias |
| `dbL0C` | L0C 双缓冲模式；当前仅值 `2` 启用，其余值均视为禁用 |
| `l1BufferStage` | A/B 的 L1 缓冲级数；`3` 启用三缓冲，其他值按双缓冲处理 |
| `groupType` | GMM tiling 兼容保留字段；当前 kernel 未读取该字段，split 方向由 `LayoutA` 对应的 `TRANS_A` 编译期路径决定（`!TRANS_A` 按 M，`TRANS_A` 按 K） |
| `groupListType` | offset、length 或 sparse |
| `singleW` | `1` 表示权重/权重 scale 是单个连续 Tensor；`0` 表示按 TensorList 逐 group 解引用 |

`kAL1/kBL1/scaleKAL1/scaleKBL1` 需满足 BlockMmad 的 L1 参数约束：`kAL1` 与 `kBL1` 的较大值作为外层 K 窗口，较大值需为较小值的整数倍；tiling 需传入一致的 `scaleKAL1` 和 `scaleKBL1`，且不小于该外层窗口，并为该外层窗口的整数倍。Kernel 使用 `scaleKAL1` 作为共享 `scaleKL1`。

### Params
```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockEpilogueParams epilogueParams;
    GM_ADDR groupListGmAddr;
    GMMTiling gmmParams;
};
```

## 特殊成员方法

### operator() 函数
```cpp
__aicore__ inline void operator()(const Params& params)
```

功能：
- 作为 kernel 入口，内部调用 `Run(params)`

### Init 函数
```cpp
__aicore__ inline void Init(const Params& params)
```

功能：
- 初始化 GM 基地址
- 读取 grouped matmul tiling 参数
- 构造首个 group 的 `problemShape_`

### Run 函数
```cpp
__aicore__ inline void Run(const Params& params)
```

功能：
- 遍历所有 group
- 逐 group 更新 `m/n/k`
- 逐 group 更新地址偏移
- 调用 scheduler 和 BlockMmad 完成单核 block 级计算

### SetMNK 函数
```cpp
__aicore__ inline void SetMNK(uint32_t groupIdx)
```

功能：
- 从 group list 中提取当前组切分值
- 当前 QGMM MX scalar 路径按 `LayoutA` 对应的 `TRANS_A` 编译期路径更新当前组的 `problemShape_`；`groupType` 在当前 kernel 中未参与该判断

### Scheduler 的 UpdateMxGroup/GetNextMxBlock 接口
```cpp
__aicore__ inline void UpdateMxGroup(const MxGroupParams& params)
__aicore__ inline bool GetNextMxBlock(MxBlockInfo& blockInfo)
```

功能：
- 仅供 GMMAQ ActivationQuant Kernel 使用
- 调度器根据当前 group 形状计算 A/B/ScaleA/ScaleB/Bias/输出基础偏移
- 调度器返回 block 形状、输入 slice 坐标及输出/输出 scale 偏移

### ProcessSingleGroup 函数
```cpp
template <bool isLastGroupAndNeedSplit>
__aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx)
```

功能：
- 普通 QGMM：Kernel 根据 Scheduler 的通用 block 坐标和自身计算的 Group 偏移构造 GM Tensor，直接交给 `BlockMmad`
- GMMAQ：Kernel 使用 Scheduler 的 MX group/block 信息，AIC 将结果写 UB，AIV 执行 Epilogue

## 调用示例

### 组件组装
```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using QgmmKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```cpp
using Params = typename QgmmKernel::Params;

Params params = {
    {m, n, k, 0},
    {aGmAddr, bGmAddr, cGmAddr, biasGmAddr, scaleAGmAddr, scaleBGmAddr},
    {},
    groupListGmAddr,
    {groupNum, m, n, k, baseM, baseN, baseK, kAL1, kBL1, scaleKAL1, scaleKBL1,
     isBias, dbL0C, l1BufferStage, groupType, groupListType, singleW}
};
```

### Kernel 执行
```cpp
QgmmKernel kernel;
kernel(params);
```

## 调度流程
```text
读取首组 tiling
    -> 初始化 block scheduler
    -> 遍历 group
    -> 更新当前组的 m/n/k
    -> 更新 A/B/Scale/Bias/C 偏移（普通 QGMM 在 Kernel 内完成；GMMAQ 由 MX Scheduler 辅助完成）
    -> scheduler 分发单核 block
    -> BlockMmad 在单核 block 内执行 L0 tile 计算
    -> 末组按需执行 tail split
```

## 适用场景
- MX 量化 grouped matmul
- group 间 `m` 或 `k` 动态变化
- 需要 sparse group list / tail split 的 grouped matmul 场景
