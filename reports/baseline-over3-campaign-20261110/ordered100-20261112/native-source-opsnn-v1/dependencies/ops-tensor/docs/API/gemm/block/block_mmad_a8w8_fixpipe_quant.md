# Block Mmad A8W8 Fixpipe Quant
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h)

## 功能说明
量化矩阵乘 Block，基于 Tensor API 实现，仅支持 AIC 计算。该组件执行由 `AType/BType` 指定的量化 A/B Cube Mmad 累加，并根据调度策略选择 L0C 输出方式：普通 QBMM 和 StreamK 的 Data Parallel（DP）block 通过 Fixpipe 反量化写入 C GM，StreamK（SK）block 将当前 K 分片的累加结果写入 workspace，等待 AIV 归约后统一反量化。文件名沿用 A8W8 路径命名，但实际输入类型不限定为 `int8_t`。

**继承自**：[Block Mmad 基础框架](./block_mmad.md)

## 特殊约束

### 调度策略限制
仅支持以下调度策略：
- `MatmulWithScaleFixpipeQuant<>`（非全载模式）
- `MatmulWithScaleFixpipeQuant<A_FULL_LOAD_MODE>`（A 矩阵全载模式）
- `MatmulWithScaleFixpipeQuant<0, true>`（Atomic Add 非全载模式）
- `MatmulWithScaleFixpipeQuant<A_FULL_LOAD_MODE, true>`（Atomic Add A 矩阵全载模式）
- `MatmulWithScaleFixpipeQuant<FullLoadMode, AtomicAdd, KernelMmadWithScaleFixpipeQuantWithoutBatch>`
  （Batch 固定为 1；`FullLoadMode` 支持非全载/A 全载，`AtomicAdd` 支持开启或关闭）
- `MatmulWithScaleFixpipeQuant<0, false, KernelQbmmPertensorMultiBlockStreamK>`（QBMM per-tensor StreamK，DP/SK 混合输出）
- `MatmulWithScaleFixpipeQuant<0, false, KernelGroupedMmadWithScaleFixpipeQuant>`（GMM Fixpipe per-channel/per-group，S8S4 等路径）
- `MatmulWithScaleFixpipeQuant<0, false, KernelGroupedMmadFixpipeQuant>`（GMM Cube Kernel，A8W8 fixpipe GMM）

`MatmulWithScaleFixpipeQuant` 的 `ScheduleType` 默认为 `KernelMmadWithScaleFixpipeQuant`（QBMM Cube Kernel 使用），可通过第 3 个模板参数 `ScheduleType_` 覆盖为：
- `KernelMmadWithScaleFixpipeQuantWithoutBatch`：仅处理 Batch 为 1 的场景，Block 的计算与输出逻辑不变，不执行多 Batch 广播和地址换算；
- `KernelQbmmPertensorMultiBlockStreamK`：QBMM per-tensor StreamK，Block 通过 `ScheduleType` 编译期判断开启 raw workspace 输出能力；
- `KernelGroupedMmadWithScaleFixpipeQuant`：GMM/S8S4 Fixpipe per-channel/per-group；
- `KernelGroupedMmadFixpipeQuant`：GMM Cube Kernel（A8W8 fixpipe GMM）。

除 `KernelQbmmPertensorMultiBlockStreamK` 外的调度都将结果写入 C GM、不启用 workspace 输出；只有它会在编译期启用原始累加结果的 workspace 输出路径。

不支持 `MatmulWithScaleMx`、`GroupedMatmulWithScaleMx` 或 `MatmulMultiBlockBasic`。

### Grouped Matmul 扩展

当 `ScheduleType` 为 `KernelGroupedMmadWithScaleFixpipeQuant` 时，本 Block 新增 GMM Fixpipe 能力：

- per-channel：完成完整 K 轴 Mmad 后，使用当前 N 分片的 scale 执行一次 Fixpipe 反量化。
- per-group：`ProcessPerGroup()` 按 `quantGroupSize` 切分 K 轴；每个 K-group 使用独立 scale 完成 Mmad 和 Fixpipe，首个 group 初始化输出，后续 group 累加到同一结果。
- per-group 循环结束后恢复完整 K 轴循环状态，保证同一 BlockMmad 实例可以继续处理后续基本块。

该扩展只处理算子侧已经准备好的 Cube 输入，不包含 INT4 到 INT8 的展开逻辑。

### 量化数据类型支持
| 数据类型 | 说明 | C0_SIZE |
|---------|------|---------|
| int8_t | A8W8 量化输入 | 由 `c0_element<AType>` 决定，通常为 32 |
| hifloat8_t | HiFloat8 量化输入 | 由 `c0_element<AType>` 决定，通常为 32 |
| fp8_e5m2_t | FP8 E5M2 量化输入 | 由 `c0_element<AType>` 决定，通常为 32 |
| fp8_e4m3fn_t | FP8 E4M3FN 量化输入 | 由 `c0_element<AType>` 决定，通常为 32 |

说明：
- A 矩阵类型由 `AType` 指定。
- B 矩阵模板参数为 `BTypeTuple`，其中第 0 个类型为 B 数据类型，第 1 个类型为 X2 scale GM 类型。
- 典型 A/B 组合包括 `int8_t/int8_t`、`hifloat8_t/hifloat8_t`、`fp8_e4m3fn_t/fp8_e4m3fn_t`、`fp8_e4m3fn_t/fp8_e5m2_t`、`fp8_e5m2_t/fp8_e4m3fn_t` 和 `fp8_e5m2_t/fp8_e5m2_t`，最终以 Tensor API Mmad 静态检查为准。
- L0C 累加类型使用显式条件映射：`int8_t` 输入映射为 `int32_t`，其他受支持输入映射为 `float`。

### Scale 因子类型
Fixpipe 反量化使用 X2 scale，支持两类输入方式：
- **Scalar scale**：`operator()` 的 `scaleGlobal` 为 `uint64_t`，Fixpipe 搬出时使用标量 scale。
- **Per-channel scale**：`scaleGlobal` 为 Tensor，Block 将当前 N 分片的 scale 搬入 L1，并在 Fixpipe 搬出时使用 L1 scale Tensor。
- **Per-group scale**：仅用于 GMM 调度；`scaleGlobal` 为 `[quantGroupNum, N]` 视图，每个 K-group 选择对应的一行 scale。

当 `CType` 为 `int32_t` 时，结果直接以累加值搬出，不使用 scale 反量化。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。

### 输出目标
普通 QBMM 以及 StreamK DP block 直接输出到 C GM：
- `CType = int32_t`：L0C 累加结果直接搬出。
- `CType = half/bfloat16_t/float`：通过 Fixpipe 搬出并应用 scalar/per-channel scale。

StreamK SK block 输出到 workspace GM：
- workspace 类型与 `L0CType` 相同；
- 搬出时不传入 scale，不执行反量化；
- 当前 K 分片的累加结果由 `BlockEpilogueQbmmPertensorStreamK` 归约后统一反量化。

### L1 切分要求
- `kAL1`：A 矩阵 L1 K 轴切分大小。
- `kBL1`：B 矩阵 L1 K 轴切分大小。
- `l1BufNum` 支持 2、3 或 4 缓冲。
- 非全载模式下，当 `l1BufNum == 2` 时支持 `kAL1` 与 `kBL1` 不同，并按 A/B 的 K-L1 大小选择复用策略。
- A 全载模式下，A 常驻 L1，`kAL1` 跟随 `kBL1`，适用于 A 分片复用收益明显的场景。
- 注意：kAL1与kBL1不相等时必须满足整数倍关系。

### Mmad 计算模式
使用默认 Mmad trait，执行量化 A/B Cube Mmad：
```
using MmadAtomT = asc::te::mmad_atom<
    asc::te::mmad_traits<
        asc::te::mmad_operation,
        asc::te::mmad_trait_default>>;

asc::te::mmad(MmadAtomT{}.with(mmadParams), c1Local, l0aLocal, l0bLocal);
```

Bias 仅在首个 K-L1/L0 迭代搬入 BT 并参与 Mmad。StreamK SK block 还要求 `kCntIndex == 0`，确保反量化前 bias 在多个 K 分片中只累加一次。需要在反量化后处理的 bias 不应传给本 Block；包含 DP block 的上层通路应在 tiling 阶段拒绝此类组合。

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| WEIGHT_NZ | B 矩阵是否为 NZ 格式 |
| TRANS_A | A 矩阵是否转置 |
| TRANS_B | B 矩阵是否转置 |
| C0_SIZE | A/B C0 对齐大小，由 `asc::te::c0_element<AType>` 推导 |
| C0_SIZE_L0C | L0C C0 对齐大小，固定为 16，定义于 `common_utils.h` |
| SCALE_BUFFER_NUM | X2 scale L1 缓冲数量，固定为 2，定义于 `common_utils.h` |
| DOUBLE_BUFFER_COUNT | A/B L1 双缓冲数量，固定为 2，定义于 `common_utils.h` |
| STREAMK_BIAS_IN_MMAD | 当前 StreamK 数据类型和 scale 组合是否支持在 Mmad 阶段处理 Bias。 |
| BIAS_IN_MMAD | Bias 是否由 BlockMmad 处理；非 StreamK 调度恒为 `true`，StreamK 调度由 `STREAMK_BIAS_IN_MMAD` 决定。 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BType | `BTypeTuple` 的第 0 个类型，表示 B 矩阵数据类型 |
| X2ScaleType | `BTypeTuple` 的第 1 个类型，表示 scale GM 类型 |
| L0CType | Mmad 累加输出类型，由 `AType` 推导 |
| MakeLayoutAL1 | A 矩阵 L1 Layout 构建器（根据 TRANS_A 选择 ZN/NZ） |
| MakeLayoutBL1 | B 矩阵 L1 Layout 构建器（根据 TRANS_B 选择 ZN/NZ） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR aGmAddr{nullptr};      // A 矩阵 GM 地址
    GM_ADDR bGmAddr{nullptr};      // B 矩阵 GM 地址
    GM_ADDR cGmAddr{nullptr};      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr{nullptr};   // Bias GM 地址
    GM_ADDR scaleAGmAddr{nullptr}; // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr{nullptr}; // B 矩阵 Scale GM 地址
    uint64_t oriK{0};              // 原始 K
    uint64_t kAL1{0};
    uint64_t kBL1{0};
    uint64_t l1BufNum{0};
    uint32_t mL0{0};
    uint32_t nL0{0};
    uint32_t kL0{0};
    QuantMode quantMode{QuantMode::DEFAULT};
    bool isBias{false};
    bool enableL0cPingPong{false};
};
```

说明：
- `scaleAGmAddr` 由 Kernel 层用于 per-tensor scale 融合，Block 主要消费传入的 scalar scale 或 `scaleBGmAddr` 对应的 per-channel scale Tensor。
- `isBias` 显式控制 Block 是否处理 bias，不通过 `biasGmAddr` 是否为空推导。
- `Init()` 对外接口入参为 `const Params&`。

## 对外接口

### Init函数

```cpp
__aicore__ inline void Init(const Params& params);
```

功能：初始化 BlockMmad 的问题规模、L0/L1 切分、scale 模式、Bias 标志及 L0C 双缓冲状态。新代码应优先使用该接口。

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| params | `const Params&` | BlockMmad 初始化参数。当前初始化过程使用 `oriK`、`kAL1`、`kBL1`、`l1BufNum`、`mL0/nL0/kL0`、`quantMode`、`isBias` 和 `enableL0cPingPong`。 |

返回值：无。

非全载模式下，`l1BufNum == 2` 时可分别使用 `kAL1/kBL1`；`l1BufNum` 为 3 或 4 时使用统一 K-L1 窗口。A 全载模式下，A 常驻 L1，适用于 A 分片复用收益明显的场景。

### 兼容Init函数

```cpp
__aicore__ inline void Init(
    const ProblemShape& problemShape,
    const BlockShape& l0TileShape,
    const uint64_t& kAL1,
    const uint64_t& kBL1,
    const uint64_t& l1BufNum,
    QuantMode quantMode,
    bool isBias,
    bool enableL0cPingPong);
```

功能：保留原多参数初始化方式的源码兼容性。该接口在内部将参数转换为 `Params` 后调用 `Init(const Params&)`，仅用于现有调用方迁移过渡，后续版本将废弃。新代码应直接构造 `Params`，现有调用方也应逐步迁移到 `Init(const Params&)`。

| 参数 | 说明 |
|------|------|
| problemShape | 原问题形状，K 维用于初始化完整 K 轴循环。 |
| l0TileShape | L0 tile 的 M/N/K 形状。 |
| kAL1/kBL1 | A/B 的 L1 K 轴切分大小。 |
| l1BufNum | L1 Buffer数量。 |
| quantMode | X2 scale 量化模式。 |
| isBias | 是否在 Block 中处理 Bias。 |
| enableL0cPingPong | 是否启用 L0C ping-pong。 |

返回值：无。

### 普通operator函数

```cpp
template <typename TensorA, typename TensorB, typename TScale, typename TensorBias, typename TensorC>
__aicore__ inline void operator()(
    TensorA gmA,
    TensorB gmB,
    TScale scaleGlobal,
    TensorBias gmBias,
    TensorC gmC,
    BlockShape singleShape);
```

功能：计算一个 M/N block。Block 将 A/B 搬入 L1/L0，在首次 K 累加中按需加入 Bias，并将 L0C 结果通过 Fixpipe 写入 C。

| 参数 | 说明 |
|------|------|
| gmA | 当前 block 的 A GM Tensor。 |
| gmB | 当前 block 的 B GM Tensor。 |
| scaleGlobal | Scalar `uint64_t` scale，或当前 N 分片对应的 per-channel scale Tensor。 |
| gmBias | 当前 N 分片对应的 Bias GM Tensor；仅在 `Params::isBias` 为 `true` 时读取。 |
| gmC | 当前 block 的 C GM Tensor。 |
| singleShape | 当前 block 的形状；普通 QBMM 路径使用其中的有效 M/N，K 轴循环由 `Init()` 初始化。 |

返回值：无。

### StreamK operator函数

```cpp
template <typename TensorA, typename TensorB, typename TScale, typename TensorBias, typename TensorC,
          typename TensorWorkspace>
__aicore__ inline void operator()(
    TensorA gmA,
    TensorB gmB,
    TScale scaleGlobal,
    TensorBias gmBias,
    TensorC gmC,
    TensorWorkspace gmWorkspace,
    BlockShape singleShape,
    int64_t kCntIndex,
    bool isSkBlock);
```

功能：处理 per-tensor StreamK 的 DP/SK block。DP block 通过 Fixpipe 输出 C；SK block 将原始累加结果写入 workspace，交由后续 AIV epilogue 归约。

| 参数 | 说明 |
|------|------|
| gmA/gmB | 当前 K 分片的 A/B GM Tensor。 |
| scaleGlobal | 当前计算使用的 per-tensor scale。 |
| gmBias | Bias GM Tensor；是否由该 Block 读取由 `BIAS_IN_MMAD` 决定。 |
| gmC | DP block 的 C GM 输出 Tensor。 |
| gmWorkspace | SK block 的原始累加结果输出 Tensor。 |
| singleShape | 当前 block 的 M/N/K 分片形状。 |
| kCntIndex | 当前 K 分片索引。 |
| isSkBlock | 当前 block 是否为 SK block。 |

约束：仅当调度类型为 `KernelQbmmPertensorMultiBlockStreamK` 时可用。`kCntIndex` 表示当前 K 分片索引，`isSkBlock` 表示当前是否为 SK block；Mmad Bias 仅在需要处理 Bias 且 `kCntIndex == 0` 时加入。

返回值：无。

### UpdateParamsForKSlice函数

```cpp
__aicore__ inline void UpdateParamsForKSlice(uint64_t curK);
```

功能：Kernel 切换到新的 GM K 分片时，刷新 BlockMmad 中依赖 K 的 L1 循环状态。A 全载模式下同时
重置 A 的 L1 缓存复用计数，避免跨 per-group A 分片复用缓存数据。

| 参数 | 说明 |
|------|------|
| curK | 当前 GM K 分片的有效 K 长度。 |

约束：仅当调度类型为 `KernelGroupedMmadWithScaleFixpipeQuant` 时可用。每次处理 K 分片前调用；
全部分片处理完成后，调用方应传入完整 K 以恢复后续 block 的循环状态。

返回值：无。

## 事件同步

| 事件 | 用途 |
|------|------|
| MTE1_MTE2 (0-3) | A/B 数据 L1 缓冲同步 |
| MTE1_MTE2 (4-5) | Bias L1 双缓冲同步 |
| FIX_MTE2 (0-1) | X2 scale L1 双缓冲同步 |
| MTE2_FIX (0-1) | scale GM 到 L1 后通知 Fixpipe |
| MTE2_MTE1 | GM 到 L1 搬运完成后触发 L1 到 L0 |
| MTE1_M | L1 到 L0 搬运完成后触发 Mmad |
| M_MTE1 | Mmad 与下一轮 L0 搬运同步 |

## 调用示例

### 组件组装
```
// 以下以 int8_t A/B 为例，可按 Tensor API 支持组合替换为其他类型。
using AType = int8_t;
using BType = int8_t;
using CType = bfloat16_t;
using BiasType = int32_t;
using X2ScaleType = uint64_t;

using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;

using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

Grouped Matmul（GMM Cube Kernel）场景只需替换 `ScheduleType_`，Block 实现完全复用：

```cpp
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
    0, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
```

### 组件初始化
```
BlockMmad blockMmad;
BlockMmad::Params params{};
params.oriK = k;
params.kAL1 = kAL1;
params.kBL1 = kBL1;
params.l1BufNum = l1BufNum;
params.mL0 = baseM;
params.nL0 = baseN;
params.kL0 = baseK;
params.quantMode = QuantMode::PERTENSOR_MODE;
params.isBias = isBias;
params.enableL0cPingPong = dbL0C;

blockMmad.Init(params);
```

### 组件执行
```
auto gmBlockA = gmA.slice(...);
auto gmBlockB = gmB.slice(...);
auto gmBlockBias = gmBias.slice(...);
auto gmBlockC = gmC.slice(...);

uint64_t scalarScale = ...;
BlockMmad::BlockShape singleShape{curM, curN, k, 0};

blockMmad(gmBlockA, gmBlockB, scalarScale, gmBlockBias, gmBlockC, singleShape);
```

## 数据流

### 存储层次
```
GM (量化 A/B) + GM (scale/bias)
    ↓
L1 (A/B 数据缓冲 + X2 scale + Bias)
    ↓
L0A/L0B (量化数据) + BT (Bias)
    ↓
L0C (由 L0CType 决定的累加类型)
    ↓
Fixpipe (scale 反量化)
    ↓
GM (C 输出)
```

### L1 缓冲布局
```
非全载：
A0|B0|A1|B1|...|Scale0|Bias0|Scale1|Bias1

A 全载：
B0|Scale0|Bias0|A|...|B1|Scale1|Bias1
```

### 执行流程
```
初始化 L1/L0 tile 和 scale/bias 缓冲
    ↓
搬运 scalar 或 per-channel scale
    ↓
搬运 Bias 到 L1（可选）
    ↓
根据 kAL1/kBL1 选择 K-L1 主循环
    ↓
K-L0 循环执行量化 Mmad
    ↓
Fixpipe 搬出并完成反量化
```

## 性能优化建议

### L1 K 轴配置
- `kAL1 == kBL1`：A/B 同步推进，控制逻辑最简单。
- `kAL1 > kBL1`：复用 A，适合 A 搬运压力较大或 A 全载收益明显的场景。
- `kBL1 > kAL1`：复用 B，适合 B 搬运压力较大或 B 数据复用较高的场景。

### L1 buffer 数量
- `l1BufNum = 2`：支持 A/B 不同 K-L1 窗口。
- `l1BufNum = 3`：使用三个 L1 buffer，相比四缓冲减少 L1 占用；K-L1 窗口统一为
  `min(kAL1, kBL1)`。
- `l1BufNum = 4`：使用四个 L1 buffer，提高数据搬运流水并行度；K-L1 窗口统一为
  `min(kAL1, kBL1)`。

### Scale 模式选择
- per-tensor scale 使用 scalar 传入，搬运开销最低。
- per-channel scale 需要将当前 N 分片 scale 搬入 L1，适合精度要求更高的场景。

### L0C 双缓冲
- `dbL0C > 1` 时启用 L0C ping-pong，可提升连续 block 搬出效率。
- `CType = int32_t` 时无反量化 scale 搬出，适合调试或保留累加结果的场景。
