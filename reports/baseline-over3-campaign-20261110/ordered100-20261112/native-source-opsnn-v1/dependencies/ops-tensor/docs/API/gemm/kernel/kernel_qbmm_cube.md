# Kernel Qbmm Cube
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_cube.h)

## 功能说明
该 Kernel 面向支持 Batch 广播的量化矩阵乘场景，仅在 AIC 上执行。Kernel 由
`BlockMmadA8W8FixpipeQuant` 与 `BlockSchedulerQuantBatchMatmulV3` 组装，完成量化 A/B 矩阵乘、
Batch 广播和 Bias 处理，并通过 Fixpipe 搬出结果、按需执行 scale 反量化。
`BlockMmadA8W8FixpipeQuant` 沿用 A8W8 路径命名，实际输入类型由 `AType/BType` 决定。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### 量化格式支持
Kernel 保留原有的 A/B、C/Bias 数据类型组合编译期校验，支持范围如下：

| AType | BType | CType | BiasType | L0C 累加类型 |
|-------|-------|-------|----------|---------------|
| `int8_t` | `int8_t` | `half` / `bfloat16_t` / `int8_t` / `int32_t` | `int32_t` | `int32_t` |
| `hifloat8_t` | `hifloat8_t` | `half` / `bfloat16_t` / `float` | `float` | `float` |
| `fp8_e4m3fn_t` / `fp8_e5m2_t` | `fp8_e4m3fn_t` / `fp8_e5m2_t` | `half` / `bfloat16_t` / `float` | `float` | `float` |

FP8 A/B 支持 E4M3FN、E5M2 的同型或混合组合。Bias 为可选输入，但 `BiasType` 是 BlockMmad 的模板参数；即使运行时不启用 Bias，也需要使用表中与输入类型匹配的 `BiasType` 实例化模板。

### 数据排布格式支持

| Layout | Kernel 编译期约束 |
|--------|-------------------|
| `LayoutA` | 支持 `nd_ext_layout_ptn` 和 `dn_ext_layout_ptn` |
| `LayoutB` | 支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn` 和 `zn_layout_ptn` |
| `LayoutC` | 仅支持 `nd_ext_layout_ptn` |

`LayoutA` 的 ND/DN 分别用于表达 A 矩阵不转置/转置的数据排布。以上不支持的 dtype 或 layout
组合会在 Kernel 模板实例化时通过 `static_assert` 给出明确的编译错误。

### Scale 因子要求
Scale 模式由 `x1QuantMode` 和 `x2QuantMode` 决定，取值来自 `QuantMode`：

Kernel 保留原有的 `ScaleGmType` 编译期类型校验，支持 `uint64_t`、`int64_t`、`bfloat16_t` 和 `float`。该校验只覆盖所有模式允许类型的并集；调用方仍须根据下表中的量化模式传入匹配的 Scale 数据类型和数据布局。

| 枚举值 | 取值 | 说明 |
|--------|------|------|
| `DEFAULT` | `0x0` | 不启用对应 scale |
| `PERTENSOR_MODE` | `0x1` | per-tensor scalar scale |
| `PERCHANNEL_MODE` | `0x2` | per-channel scale |
| `PERTOKEN_MODE` | `0x4` | per-token scale |
| `MX_PERGROUP_MODE` | `0x8` | MX per-group scale |
| `PERBLOCK_MODE` | `0x10` | per-block scale |
| `PERGROUP_MODE` | `0x20` | per-group scale |

`x1QuantMode` 描述 A 矩阵的 scale 模式，`x2QuantMode` 描述 B 矩阵的 scale 模式。当前 Kernel 支持以下组合：

| A 矩阵模式 (`x1QuantMode`) | B 矩阵模式 (`x2QuantMode`) | 适用场景与 Scale 处理 |
|----------------------------|----------------------------|-----------------------|
| `DEFAULT` | `DEFAULT` | A、B 均不提供反量化 scale。用于 `CType = int32_t` 的场景，Block 将 L0C 累加结果直接写入 GM。 |
| `DEFAULT` | `PERCHANNEL_MODE` | A 不提供 scale，B 的每个 N 通道分别使用一个 scale。`scaleBGmAddr` 指向按 `uint64_t` 位模式编码的 per-channel scale 数组；Kernel 根据当前 block 负责的 N 范围截取对应 scale Tensor 并传给 Block，Block 将 scale 搬入 L1，Fixpipe 搬出结果时按通道完成反量化。`int64_t` 使用相同的 64-bit 位模式。 |
| `DEFAULT` | `PERTENSOR_MODE` | A 不提供 scale，整个 B 矩阵共用一个 scalar scale。Kernel 从 `scaleBGmAddr` 读取该值，将其转换并封装为 Fixpipe 使用的 `uint64_t scaleScalar_`，再传给 Block。支持的 scale 存储类型为 `uint64_t/int64_t`、`bfloat16_t` 或 `float`。Block 在 Fixpipe 搬出结果时使用该 scalar scale 完成反量化。 |
| `PERTENSOR_MODE` | `PERTENSOR_MODE` | A、B 各自提供一个 per-tensor scalar scale，二者的存储类型均为 `float`。Kernel 分别从 `scaleAGmAddr` 和 `scaleBGmAddr` 读取 scale，计算两者乘积，将乘积转换并封装为 Fixpipe 使用的 `uint64_t scaleScalar_`，再传给 Block。Block 在 Fixpipe 搬出结果时使用该乘积 scale 完成反量化。 |

其他组合当前模板未提供对应 scale 数据流。

当输出 `CType = int32_t` 时，无论传入 Block 的是 scalar scale 还是 per-channel scale，Block 都不会使用 scale，也不会执行 Fixpipe 反量化，而是将 L0C 中的 `int32_t` 累加结果直接写入 GM。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算（AIV 核在 `Init` 中直接返回）。

### BlockMmad 限制
仅支持调度策略为 `MatmulWithScaleFixpipeQuant` 的 `BlockMmad`，即 `BlockMmad::DispatchPolicy::ScheduleType` 必须为 `KernelMmadWithScaleFixpipeQuant`。

### BlockScheduler 限制
使用 `BlockSchedulerQuantBatchMatmulV3` 调度器，支持 block 切分、尾块切分和多 Batch 维度遍历。

### Batch 维度限制
支持 4 维 Batch（`batchA1/A2/A3/A4`、`batchB1/B2/B3/B4`、`batchC1/C2/C3/C4`）。多 Batch 路径按 C 的 Batch 空间遍历，并根据 A/B 到 C 的广播倍数计算 GM 地址偏移。

### Atomic Add 模式
可选 Atomic Add 模式（`IS_ATOMIC_ADD = true`）。Kernel 在计算开始时配置 atomic add，结束时恢复 atomic none。

## 模板参数

### 模板定义
```
template <
    class ProblemShape,      // 问题形状类型
    class BlockMmad,         // Fixpipe Quant BlockMmad
    class BlockEpilogue,     // 后处理组件（通常为空）
    class BlockScheduler>    // BlockSchedulerQuantBatchMatmulV3 调度器
```

### 模板参数说明
| 参数 | 说明 |
|------|------|
| ProblemShape | 问题形状类型，包含 m、n、k、b |
| BlockMmad | Fixpipe Quant BlockMmad，基于 `MatmulWithScaleFixpipeQuant` |
| BlockEpilogue | 后处理组件，当前路径通常使用 `BlockEpilogueEmpty` |
| BlockScheduler | QBMM V3 调度器 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BlockMmadParams | BlockMmad 参数类型，包含 GM 地址 |
| AType/BType/CType/BiasType | 继承自 BlockMmad 的数据类型 |
| X2ScaleType | Kernel 内部使用的 scalar scale 类型，固定为 `uint64_t` |
| ScaleGmType | BlockMmad 中 `BTypeTuple` 的第 1 个类型；支持 `uint64_t`、`int64_t`、`bfloat16_t` 和 `float` |
| WEIGHT_NZ | B 矩阵是否为 NZ 格式（继承自 BlockMmad） |
| TRANS_A | A 矩阵是否转置（继承自 BlockMmad） |
| TRANS_B | B 矩阵是否转置（继承自 BlockMmad） |
| IS_ATOMIC_ADD | 是否启用 Atomic Add 模式（继承自 BlockMmad::DispatchPolicy） |
| C0_SIZE | A/B C0 对齐大小 |
| MakeLayoutA/MakeLayoutB/MakeLayoutC | A/B/C GM Tensor Layout 构建器 |

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;      // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;     // BlockMmad 参数（包含 GM 地址）
    BlockSchedulerParams schParams; // scheduler 参数
    QBMMTiling qbmmParams;          // QBMM Cube 参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4; // A 矩阵 Batch 维度
    uint32_t batchB1, batchB2, batchB3, batchB4; // B 矩阵 Batch 维度
    uint32_t batchC1, batchC2, batchC3, batchC4; // C 矩阵 Batch 维度
    uint32_t biasThreeDim;                       // Bias 是否按 Batch 展开
    uint32_t x1QuantMode;                        // x1 scale 量化模式
    uint32_t x2QuantMode;                        // x2 scale 量化模式
    uint32_t kAL1;                               // A 的 L1 K 轴切分
    uint32_t kBL1;                               // B 的 L1 K 轴切分
    uint32_t nBufferNum;                         // A/B L1 缓冲数量
    uint32_t baseM, baseN, baseK;                // L0 tile 形状
    uint32_t isBias;                             // 是否启用 bias
    uint32_t dbL0C;                              // L0C 双缓冲标志
    uint32_t bMustHitL2 = 1U;                    // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

### BlockMmadParams
```
struct Params {
    GM_ADDR aGmAddr;      // A 矩阵 GM 地址
    GM_ADDR bGmAddr;      // B 矩阵 GM 地址
    GM_ADDR cGmAddr;      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;   // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr; // x1 scale GM 地址
    GM_ADDR scaleBGmAddr; // x2 scale GM 地址
    uint64_t oriK;
    uint64_t kAL1;
    uint64_t kBL1;
    uint64_t l1BufNum;
    uint32_t mL0;
    uint32_t nL0;
    uint32_t kL0;
    QuantMode quantMode;
    bool isBias;
    bool enableL0cPingPong;
};
```

## 对外接口

### 构造函数

```cpp
__aicore__ inline GemmUniversal();
```

功能：构造 QBMM Cube Kernel 对象。

### 析构函数

```cpp
__aicore__ inline ~GemmUniversal();
```

功能：析构 QBMM Cube Kernel 对象。

### operator函数

```cpp
__aicore__ inline void operator()(const Params& params);
```

功能：执行完整的 QBMM Cube Kernel 计算。该方法是 Kernel 唯一对外开放的执行接口，内部完成组件初始化、Batch 广播、Block 调度、Tensor 切片及 BlockMmad 调用。

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| params | `const Params&` | Kernel 参数，包含问题规模、BlockMmad 参数、BlockScheduler 参数和 QBMM tiling 参数。 |

返回值：无。

约束：

- `qbmmParams.isBias` 为非零值时启用 Bias，`mmadParams.biasGmAddr` 必须指向有效的 Bias 数据。

## 执行流程

Kernel 对外通过 `operator()` 执行完整 QBMM Cube 计算，内部流程概括如下：
1. 解析 `Params`，设置 A/B/C/Bias/Scale 的 GM 地址和 scale 模式。
2. 按 `baseM/baseN/baseK/kAL1/kBL1/nBufferNum/x2QuantMode/isBias/dbL0C` 初始化 BlockMmad。
3. 由 `BlockSchedulerQuantBatchMatmulV3` 负责 block 坐标、尾块和 Batch 遍历。
4. 单 Batch 场景直接遍历 block；多 Batch 场景按 C 的 4 维 Batch 空间遍历，并根据广播关系更新 A/B/C 地址偏移。
5. per-channel scale 场景向 BlockMmad 传入当前 N 分片 scale Tensor；per-tensor scale 场景传入折算后的 scalar scale。
6. Atomic Add 模式下，Kernel 在计算前后分别配置和恢复 atomic 状态。

## 调用示例

### 组件组装
```
// 以下以 int8_t A/B 为例；替换为 hifloat8_t 或 fp8 类型时，CType/BiasType 也必须使用支持表中的组合。
using AType = int8_t;
using BType = int8_t;
using CType = bfloat16_t;
using BiasType = int32_t;
using X2ScaleType = uint64_t;

using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, 0, LayoutA, LayoutB, AType>;

using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```
using Params = typename QBMMKernel::Params;
Params params;

params.problemShape = {m, n, k, batch};
params.mmadParams.aGmAddr = aGM;
params.mmadParams.bGmAddr = bGM;
params.mmadParams.cGmAddr = cGM;
params.mmadParams.biasGmAddr = biasGM;
params.mmadParams.scaleAGmAddr = pertokenScaleGM;
params.mmadParams.scaleBGmAddr = scaleGM;

params.schParams = {
    baseM, baseN,
    mTailTile, nTailTile,
    mBaseTailSplitCnt, nBaseTailSplitCnt,
    mTailMain, nTailMain
};

params.qbmmParams = {
    batchA1, batchA2, batchA3, batchA4,
    batchB1, batchB2, batchB3, batchB4,
    batchC1, batchC2, batchC3, batchC4,
    biasThreeDim,
    x1QuantMode, x2QuantMode,
    kAL1, kBL1, nBufferNum,
    baseM, baseN, baseK,
    isBias, dbL0C, bMustHitL2
};
```

### Kernel 执行
```
QBMMKernel kernel;
kernel(params);
```

## 数据流

### 单 Batch 流程
```
Params
    ↓
解析 GM 地址和 scale
    ↓
BlockScheduler 获取 block
    ↓
Slice A/B/C/Bias/Scale
    ↓
BlockMmadA8W8FixpipeQuant
    ↓
GM 输出
```

### 多 Batch 流程
```
4 维 Batch C 空间遍历
    ↓
按广播倍数计算 A/B Batch 偏移
    ↓
更新 A/B/C GM 地址偏移
    ↓
进入单 Batch block 处理流程
```

### Scale 处理流程

Scale 的处理分为 Kernel 准备和 Block 搬出两个阶段。Kernel 根据 `x1QuantMode/x2QuantMode` 确定 scale 的来源和形态，并将 scalar scale 或当前 N 分片的 per-channel scale Tensor 传给 Block。Mmad 计算始终先在 L0C 中完成累加；仅当输出类型不是 `int32_t` 时，Block 才在结果搬出阶段使用 scale 进行 Fixpipe 反量化。

- `CType = int32_t`：Mmad 完成后，Block 将 L0C 中的 `int32_t` 累加结果直接写入 GM。该路径不使用传入 Block 的 scale，不执行 Fixpipe 反量化，也不把累加结果转换为浮点类型。
- 对支持表中的非 `int32_t` 输出类型，且 B 使用 per-channel scale：Kernel 根据当前 block 的 N 方向起始位置和长度，从 `scaleBGmAddr` 中取得对应通道的 scale Tensor。Block 将该 Tensor 从 GM 搬入 L1，Fixpipe 搬出 L0C 结果时对每个 N 通道应用各自的 scale，并转换为 `CType` 后写入 GM。
- 对支持表中的非 `int32_t` 输出类型，且仅 B 使用 per-tensor scale：Kernel 从 `scaleBGmAddr` 读取整个 B 矩阵共用的 scalar scale，将其转换并封装为 `uint64_t scaleScalar_` 后传给 Block。Fixpipe 搬出 L0C 结果时对所有元素使用同一个 scale，并转换为 `CType` 后写入 GM。
- 对支持表中的非 `int32_t` 输出类型，且 A、B 均使用 per-tensor scale：Kernel 从 `scaleAGmAddr` 和 `scaleBGmAddr` 分别读取 A、B 的 `float` scalar scale，先计算二者的乘积，再将乘积转换并封装为 `uint64_t scaleScalar_` 后传给 Block。Fixpipe 使用该乘积 scale 完成反量化和输出类型转换，然后将结果写入 GM。

## 性能优化建议

### Block 与 L1 配置
- `baseM/baseN/baseK` 应与 BlockMmad 的 L0/L1 切分匹配。
- `kAL1/kBL1` 可根据 A/B 复用关系配置；详见 [Block Mmad A8W8 Fixpipe Quant](../block/block_mmad_a8w8_fixpipe_quant.md)。
- `nBufferNum = 4` 侧重提高搬运流水并行度，`nBufferNum = 3` 相比四缓冲减少 L1 占用，
  `nBufferNum = 2` 支持 A/B 使用不同的 K-L1 窗口。

### Batch 广播
- 多 Batch 场景下，A/B/C 的 4 维 Batch 参数需满足广播关系。
- `biasThreeDim = 1` 时，Bias 随 C Batch 偏移；否则所有 Batch 复用同一 Bias。

### 尾块切分
- 尾轮由 scheduler 更新尾块切分，用于提升尾块场景下的核利用率。
- `mTailTile/nTailTile` 应与 scheduler 参数保持一致。
