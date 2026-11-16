# Block Mmad Weight Prologue MX
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_weight_prologue_mx.h)

## 功能说明

这是 `BlockMmad<MatmulWithWeightQuantMx, ...>` 的 arch35 特化，负责 AIC（Cube）侧的单 tile
MX MMAD。B 和 bias 由 Kernel 内的 AIV（Vector）前处理写入共享 L1，因此本 BlockMmad 不接收
B/Bias Tensor，也不执行 B GM→L1 搬运。

**继承自**：[BlockMmad 基础框架](./block_mmad.md)

**配套组件**：[Kernel Matmul Mix Weight Prologue](../kernel/kernel_matmul_mix_weight_prologue.md) 和
[Weight Quant MX 前处理 Tile](../tile/tile_weight_quant_mx_preprocess.md)

## 特殊约束

### 架构和 policy

仅支持 `__NPU_ARCH__ == 3510`（arch35），且只能与 `MatmulWithWeightQuantMx` 组合。

### 模板类型约束

- `ATypeTuple`、`LayoutATuple`、`BTypeTuple`、`LayoutBTuple` 必须都是二元 tuple。
- Block 的静态约束要求 A 类型占 1 字节；当前配套 prologue/算子组装固定为 `fp8_e4m3fn_t`。
- B 类型必须为 packed FP4，ScaleA/ScaleB 必须为 `fp8_e8m0_t`。
- `LayoutB` 必须是转置布局：NZ 使用 `zn_layout_ptn`，ND 使用 `dn_ext_layout_ptn`。
- 当前算子组装中 `LayoutC`/`LayoutBias` 均为 `nd_ext_layout_ptn`；bias 可选，有 bias 时
  `BiasType` 与 `CType` 一致。

### 缓冲和同步

- 调用方必须传入 2 或 4 个 L1 buffer；配套 host tiling 仅生成这两种值，Block 不重复校验。
- AIV 是 B/bias L1 的生产者，AIC 等待 ready 标志；AIC 完成当前 K window 后发送 free 标志。
- A、ScaleA、ScaleB 的 L1/L0 搬运由本组件负责；B/bias 的 L1 地址和槽位所有权来自
  `WeightL1Storage` 协作契约。

## 模板类型和别名

```cpp
template <
    class ATypeTuple, class LayoutATuple, class BTypeTuple, class LayoutBTuple,
    class CType, class LayoutC, class BiasType, class LayoutBias>
class BlockMmad<MatmulWithWeightQuantMx, ATypeTuple, LayoutATuple, BTypeTuple, LayoutBTuple,
                CType, LayoutC, BiasType, LayoutBias>;
```

| 模板参数 | 公开类型别名 |
| :--- | :--- |
| `ATypeTuple` | `AType`、`ScaleAType` |
| `LayoutATuple` | `LayoutA`、`LayoutScaleA` |
| `BTypeTuple` | `BType`、`ScaleBType` |
| `LayoutBTuple` | `LayoutB`、`LayoutScaleB` |
| 独立类型/layout 参数 | `CType`、`LayoutC`、`BiasType`、`LayoutBias` |

## 特殊数据结构

### `Params`

```cpp
struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR scaleAGmAddr;
    GM_ADDR scaleBGmAddr;
    GM_ADDR cGmAddr;
    L1TileShape l1TileShape;
    L0TileShape l0TileShape;
    uint64_t l1BufferNum;
    bool hasBias;
};
```

| 字段 | 说明 |
| :--- | :--- |
| `aGmAddr` | A 矩阵 GM 地址 |
| `scaleAGmAddr` / `scaleBGmAddr` | ScaleA/ScaleB GM 地址 |
| `cGmAddr` | 输出 C GM 地址 |
| `l1TileShape` | `(baseM, baseN, kL1, scaleKL1)` |
| `l0TileShape` | `(baseM, baseN, baseK)` |
| `l1BufferNum` | L1 B/A/bias buffer 数，支持 2 或 4 |
| `hasBias` | 是否消费 AIV 写入的 bias |

### 协作接口

`SyncProtocol` 提供 ready/free 标志的模式和编号；`WeightL1Storage` 提供 `Init`、
`MakeWeightTensor`、`MakeBiasTensor`、`WeightBufferSize`、`BiasBufferSize` 和 `BuffersPerHalf`。
它们是 Kernel prologue 与 BlockMmad 之间的内部协作契约，调用方通常通过 Kernel 使用，不应自行创建
第二套 L1 地址或同步协议。

## 特殊成员方法

### 构造函数和析构函数

```cpp
__aicore__ inline explicit BlockMmad(const Params& params);
__aicore__ inline ~BlockMmad();
```

构造函数初始化 L1 存储规划和各级缓冲槽位；析构函数关闭 MM layout transform。

### `operator()` 函数

```cpp
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void operator()(
    const TensorA& tensorA, const TensorScaleA& tensorScaleA,
    const TensorScaleB& tensorScaleB, const TensorC& tensorC);
```

输入 Tensor 应为当前 scheduler tile 的 GM slice。B/bias 不作为参数传入，而由 AIV 通过共享 L1 提供。

## 计算流程

1. 在 AIC 侧将 A、ScaleA、ScaleB 搬入 L1。
2. 在 L1 lock 保护下并行搬运 A 并清理 `[Align32(K), Align64(K))` 的 B 物理尾部。
3. 等待 AIV 为当前 L1 weight buffer 设置 ready 标志。
4. 按 `baseK` 将 A、B、ScaleA、ScaleB 搬入 L0，使用 `MmadTraitMX` 累加到 L0C。
5. 首个 K tile 消费 AIV 已按 `1/64` 缩放的 bias，Fixpipe 将结果写回 C。
6. 释放 weight buffer，允许 AIV 继续生产下一 tile。

## Kernel 内部调用

该 Block 不能脱离 AIV producer 独立调用。构造时会向 AIV 释放 weight buffer，执行时会等待 AIV
按相同 tile 序列写入 B/bias 并发送 ready 标志。完整类型组装和参数构造见
[Kernel Matmul Mix Weight Prologue](../kernel/kernel_matmul_mix_weight_prologue.md#调用示例)。

```cpp
BlockMmad blockMmad(params.mmadParams);
// AIV prologue and AIC must consume the same scheduler tile before this call completes.
blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, gmBlockC);
```

## 适用场景

- `GemmUniversal` Weight Prologue 特化的 AIC 计算侧。
- 配套 Kernel 支持 arch35 MXA8W4 Weight ND/NZ 输入，并由 AIV 负责 packed FP4→FP8 和 bias 前处理。
- 不适合单独调用；没有对应 AIV producer 时，B/bias L1 数据和 ready 标志不存在。
