# Kernel QGMM SwiGLU MX

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h)

## 功能说明

`KernelGmmSwiGluMixMx` 对应的 `GemmUniversal` 特化将 MXFP8 Grouped
MatMul、SwiGLU 和 MX 输出量化组成一个 AIC:AIV=1:2 的 MIX Kernel。输入矩阵乘
的完整 N 维由前后两个等宽半区组成，AIC 计算两个半区，AIV 执行 SwiGLU 后将
输出宽度缩减为原来的一半，并写出量化结果 `y` 与 E8M0 `yScale`。

本组件使用
[BlockMmad QGMM MX](../block/block_mmad_qgmm_mx.md)、
[BlockScheduler GMM SWAT](../block/block_scheduler_gmm_swat_with_tail_split.md)
和 [BlockEpilogue SwiGLU MX Quant](../../epilogue/block/block_epilogue_swiglu_mx_quant.md)。

## 支持范围

- `AType`、`BType` 仅支持 MXFP8；`CType`、`BiasType` 固定为 `float`。
- `LayoutA` 固定为 ND。
- `LayoutB` 支持 ND、DN、NZ 和 ZN；NZ/ZN 为物理 WeightNZ 路径。
- `LayoutC`、`LayoutBias` 不支持 NZ/ZN。
- ScaleA、ScaleB 和输出 scale 使用 E8M0。
- 当前融合路径不计算 bias，`cGmAddr` 和 `biasGmAddr` 为未使用槽位。
- `groupType` 固定为 0；`groupListType` 支持 0（累计 offset）和 1（逐组 count）。
- V3 WeightNZ 接口使用 `singleW=1`，即一个连续 Tensor 承载全部 expert。

## Shape 与布局契约

下表使用 `NFull=2*NOut`。源码中的 `GMMTiling::n` 和 `ProblemShape::N`
均表示 `NFull`。

| 张量 | 逻辑 shape | dtype / layout | 空指针语义 |
| --- | --- | --- | --- |
| A / X | `[M, K]` | MXFP8 / ND | `groupNum>0` 时必须非空 |
| B / Weight（非转置） | `[E, K, NFull]` | MXFP8 / ND 或 NZ | 必须非空 |
| B / Weight（转置） | `[E, NFull, K]` | MXFP8 / DN 或 ZN | 必须非空 |
| ScaleA / XScale | `[M, ceil(K/64), 2]` | E8M0 / ScaleA-ND | 必须非空 |
| ScaleB / WeightScale（非转置） | `[E, ceil(K/64), NFull, 2]` | E8M0 / ScaleB-ND | 必须非空 |
| ScaleB / WeightScale（转置） | `[E, NFull, ceil(K/64), 2]` | E8M0 / ScaleB-DN | 必须非空 |
| groupList | `[E]` | int64 / ND | `groupNum>0` 时必须非空 |
| y | `[M, NOut]` | MXFP8 / ND | 必须非空 |
| yScale | `[M, ceil(NOut/64), 2]` | E8M0 / ND | 必须非空 |

表中转置 shape 是低层 Kernel 接收的归一化内部 Tensor。上层 ACLNN
WeightNZ V3 接口通过 stride 识别 ZN，其公开入口 view 仍为 `[E,K,NFull]` 和
`[E,ceil(K/64),NFull,2]`；上层在下发 Kernel 前分别交换 K/N 与 scaleK/N，形成
表中的转置内部 shape。直接调用 Blaze Kernel 的示例则按表中 shape 构造 ZN/DN Tensor。

NZ/ZN 权重按 expert 分别做物理 padding。Kernel 依据实际 `K/NFull` 的分形
大小为 `singleW=1` 计算每个 expert 的物理起始地址，不能使用未 padding 的逻辑
元素数作为 expert stride。共享 Kernel 的 `singleW=0` TensorList 路径直接使用
对应 expert 的 Tensor 地址，不再追加 group stride；这不扩展 V3 WeightNZ 的
单 Tensor 限制。ScaleB 仍使用 ScaleB-ND/ScaleB-DN 逻辑布局。

## SwiGLU 与量化口径

共享 Epilogue 保留 `swigluMode=0` 以兼容既有 V2 内部调用；本 V3 路径只允许
`swigluMode=2`，不公开 mode 0、mode 1 或其他模式。mode 2 将前半区作为激活
分支，后半区作为 gate：

```text
act  = min(left, clampLimit)
gate = clamp(right, -clampLimit, clampLimit) + gluBias
yFp  = act / (1 + exp(-gluAlpha * act)) * gate
```

V3 MXFP8 接口公开 `scaleAlg=0`（OCP）和 `scaleAlg=1`（cuBLAS），并要求
`dstTypeMax=0`。Epilogue 复用 `Tile::MxQuant` 并只选择 OCP/cuBLAS 算法，
没有动态 dtype range 分支，V3 调用方不得传入 `scaleAlg=2`。

## 参数结构

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
    uint8_t dbL0C;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
};

struct Params {
    ProblemShape problemShape;
    BlockMmadAddressParams blockMmadAddressParams;
    BlockEpilogueParams epilogueParams;
    GM_ADDR groupListGmAddr;
    GMMTiling gmmParams;
};
```

`gmmParams` 是运行时问题规模的权威来源；`problemShape` 应与其
`m/n/k` 保持一致。当前 BlockMmad 使用 `scaleKAL1` 作为共享
`scaleKL1`，因此 `scaleKBL1` 必须与 `scaleKAL1` 相等。

## NZ/ZN 双源搬运

对每个输出 N tile，Kernel 分别建立左右半区的 Weight 和 WeightScale slice。
NZ/ZN 路径将四个 slice 显式传给 BlockMmad，BlockMmad 在 L1 中拼成
`[left, right]` 后执行 MMAD；ND/DN 兼容路径继续使用单源拼接接口。优化搬运
条件不满足时会回退到两个独立 Tensor copy，逻辑结果不变。

## 调用与校验边界

```cpp
Kernel kernel;
kernel(params);
```

Blaze Kernel 是返回 `void` 的设备侧模板，不负责 ACLNN 状态码，也不会返回
`161001`。必选地址、TensorList、shape、dtype、layout 和属性合法性必须由
上层 ACLNN/tiling 在下发 Kernel 前校验；上层对必选空指针按接口规范返回
`161001`。
