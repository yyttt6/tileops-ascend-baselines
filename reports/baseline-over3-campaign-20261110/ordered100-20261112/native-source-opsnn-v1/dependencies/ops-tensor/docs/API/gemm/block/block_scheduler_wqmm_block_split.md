# Block Scheduler Wqmm Block Split
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_wqmm_block_split.h)

## 功能说明

固定核分核（fixed-core）的 Matmul 调度器：构造时读取当前核 `blockIdx`，把 M/N 平面按
`cubeNumBlocksM × cubeNumBlocksN` 划分为责任矩形，每核固定负责一个
`singleCoreM × singleCoreN` 矩形；随后三个查询接口把该矩形展开为核内局部 tile 序列，
并支持 ORDER_M/ORDER_N 两种核内 swizzle 遍历顺序。

与 SWAT 系调度器的区别在分核时机：SWAT 描述全局 tile、由 Kernel 按 `blockIdx` 跨核步进；
本调度器在构造期完成分核，Kernel 只用局部 `tileIdx ∈ [0, GetBlockNums())` 驱动循环。AIC 与
其配对的 AIV 构造出相同的 tile 序列，两侧逐 tile 对齐。

**配套组件**：[Kernel Wqmm Mix Pergroup](../kernel/kernel_wqmm_mix_pergroup.md)

## 特殊约束

### 分核与 AIV 折算

构造期通过 `GetCurrentBlockIdx()` 获取逻辑 block id，内部完成 AIV 的 taskRation 折算：
AIC 直接取 `GetBlockIdx()`，AIV 取 `GetBlockIdx() / GetTaskRation()`。因此同一 QUAD 内
AIC 和两个 AIV 得到相同的逻辑 id、相同的责任矩形和相同的 tile 序列。

逻辑 id 越界（超过 `cubeNumBlocksM * cubeNumBlocksN`）或责任矩形起点越过 M/N 边界时，
`GetBlockNums()` 返回 0，Kernel 直接空转返回。

### 对齐

- M 方向核间对齐粒度为 `MATMUL_MNK_ALIGN`（16）。
- N 方向：Weight NZ 且非转置时按 `c0_element<AType>` 对齐（FP8 为 32，满足 fractal 寻址
  的 32 对齐不变式），其余按 `BLOCK_CUBE`（16）。

### tiling 不变式

依赖 host tiling 固定填 `stepM == stepN == 1`（L1 组 == L0 tile），两级 L1/L0 遍历退化为
单级；本调度器不再表达 L1 级步进参数。

## 模板参数

```cpp
template <class ProblemShape_, class LayoutB_, class AType_>
class BlockSchedulerWqmmBlockSplit;
```

| 参数 | 说明 |
| :--- | :--- |
| `ProblemShape_` | `asc::te::shape<int64_t, int64_t, int64_t>`，维序 `(M, N, K)` |
| `LayoutB_` | B 矩阵 layout pattern，用于判断 NZ 格式与转置（决定 N 方向核间对齐粒度） |
| `AType_` | A 矩阵数据类型，NZ 对齐粒度按元素宽度选择 |

| 公开类型别名 | 说明 |
| :--- | :--- |
| `BlockCoord` | `asc::te::coord<int64_t, int64_t, int64_t, int64_t>`，`(mOffset, nOffset, 0, 0)` 为 tile 原点的 GM 绝对元素坐标 |
| `BlockShape` | `asc::te::shape<int64_t, int64_t, int64_t, int64_t>`，`(validM, validN, kSize, 1)` |

## 特殊数据结构

### `Params`

```cpp
struct Params {
    uint32_t cubeNumBlocksM{0};
    uint32_t cubeNumBlocksN{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t iterateOrder{0};
};
```

| 字段 | 说明 |
| :--- | :--- |
| `cubeNumBlocksM` / `cubeNumBlocksN` | M/N 方向的责任矩形数，乘积为参与计算的核数 |
| `baseM` / `baseN` | 单 tile 的 M/N 基准大小（来自 host tiling 的 baseM/baseN） |
| `iterateOrder` | 核内遍历顺序：`ORDER_N`（M 快变）或 `ORDER_M`（N 快变），与 ops-nn 旧实现的 iterateOrder 语义一致 |

## 特殊成员方法

### 构造函数

```cpp
__aicore__ inline BlockSchedulerWqmmBlockSplit(const ProblemShape& problemShape, const Params& params);
```

完成全部 per-core 初始化：读取问题规模与 `baseM/baseN`，按逻辑 block id 计算
`mDimIdx/nDimIdx`，得到责任矩形起点 `coreMOffset/coreNOffset` 与实际大小
`singleCoreM/singleCoreN`，最后按 `baseM/baseN` 切出 `mTileCount_ * nTileCount_` 个 tile。
构造之后调度器不再有可变状态。

### `GetBlockNums`

```cpp
__aicore__ inline uint64_t GetBlockNums() const;
```

返回当前核责任矩形内的 tile 数；越界核返回 0。

### `GetBlockCoord`

```cpp
__aicore__ inline BlockCoord GetBlockCoord(uint64_t tileIdx) const;
```

`tileIdx` 是当前核内的局部序号；返回 tile 原点的 GM 绝对元素坐标。`iterateOrder ==
ORDER_N` 时 M 快变（先走完一列 M tile 再进位 N），否则 N 快变——两种顺序对应 ops-nn 旧
`IterMatmulOut` 的两条遍历分支。

### `GetBlockShape`

```cpp
__aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord) const;
```

按坐标裁剪当前 tile 的尾块：`validM/validN` 取 `baseM/baseN` 与责任矩形剩余量的较小值；
K 维返回完整 `kSize`，K 方向的分段由 BlockMmad 内部循环负责。

## Kernel 内部调用

```cpp
BlockScheduler scheduler(params.problemShape, params.schedulerParams);
const uint64_t tileCount = scheduler.GetBlockNums();
for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
    const auto blockCoord = scheduler.GetBlockCoord(tileIdx);
    const auto blockShape = scheduler.GetBlockShape(blockCoord);
    // AIC: blockMmad(blockA, blockYScale, blockC, blockShape);
    // AIV: blockPrologue(gmBlockB, gmBlockScale, blockShape);
}
```

AIC 与 AIV 必须使用相同的 `ProblemShape` 和 Scheduler 参数构造，保证两侧按同一个
tile 序列工作。完整用法见 [Kernel Wqmm Mix Pergroup](../kernel/kernel_wqmm_mix_pergroup.md#执行流程)。

## 适用场景

- T-CG per-group A8W4 量化 matmul（`KernelMixWithWeightPergroupPrologue` 调度类型）的
  AIC/AIV 混合调度，每核责任区固定为一个 M/N 矩形。
- 需要与 ops-nn 旧实现逐 tile 对齐（相同分核、相同遍历顺序）的迁移场景。
