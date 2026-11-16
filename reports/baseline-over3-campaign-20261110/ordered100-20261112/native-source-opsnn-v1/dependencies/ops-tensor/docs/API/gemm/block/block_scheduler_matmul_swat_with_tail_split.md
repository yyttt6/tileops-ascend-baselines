# Block Scheduler Matmul SWAT With Tail Split
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h)

## 功能说明

`BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>` 是无量化语义的 M/N SWAT Scheduler。
它按四行滑动窗口扫描基本 tile，在奇数窗口反转 N 方向，并将尾块 split 压缩为连续 tile index，
避免无效 split 占用核。

**配套组件**：[Kernel Matmul Mix Weight Prologue](../kernel/kernel_matmul_mix_weight_prologue.md)
和 [Block Scheduler 基础框架](./block_scheduler.md)

## 特殊约束

- `ProblemShape` 的维序为 `(M, N, K)`，Scheduler 不改变 K，返回的 `BlockShape` 为
  `(curM, curN, K, 1)`。
- `baseM`、`baseN` 必须非零；调用方必须保证 tile shape、split 参数与 AIC/AIV 消费协议一致。
- Scheduler 不携带量化、权重格式或 bias 语义；ND/NZ 由 Kernel 的 `LayoutB` 处理。
- AIC 和 AIV 必须使用同一组 `Params`、同一 `tileIdx` 序列。

## 特殊数据结构

### `Params`

```cpp
struct Params {
    uint64_t baseM;
    uint64_t baseN;
    uint64_t mTailTile;
    uint64_t nTailTile;
    uint64_t mBaseTailSplitCnt;
    uint64_t nBaseTailSplitCnt;
    uint64_t mTailMain;
    uint64_t nTailMain;
};
```

| 字段 | 说明 |
| :--- | :--- |
| `baseM` / `baseN` | 基本 M/N tile 大小 |
| `mTailTile` / `nTailTile` | 最后一轮尾块的 M/N split 数 |
| `mBaseTailSplitCnt` / `nBaseTailSplitCnt` | 尾部合并区域包含的基本 tile 数 |
| `mTailMain` / `nTailMain` | split 区域中非最后一块的逻辑尺寸 |

### `BlockCoord` 和 `BlockShape`

`BlockCoord` 的字段依次为 `(mOffset, nOffset, logicalTileIdx, splitIdx)`；`splitIdx == -1`
表示普通 tile。`BlockShape` 的字段依次为 `(curM, curN, K, 1)`。

## 特殊成员方法

### 构造函数

```cpp
__aicore__ inline BlockSchedulerMatmulSwatWithTailSplit(
    const ProblemShape& problemShape, const Params& params);
```

构造函数计算基本 M/N tile 数、窗口信息和紧凑尾块数量。

### 获取 tile 信息

```cpp
__aicore__ inline uint64_t GetTileCount() const;
__aicore__ inline BlockCoord GetBlockCoord(uint64_t tileIdx) const;
__aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord) const;
```

`GetTileCount()` 返回压缩后的有效 tile 数；`GetBlockCoord()` 不返回零大小 split；
`GetBlockShape()` 根据 split index 计算当前 tile 的有效 M/N。

## 调度流程

1. 根据 `(M, N)` 和 `baseM/baseN` 计算基本 tile 网格。
2. 使用四行窗口扫描 M 方向；奇数窗口反转 N 方向，改善相邻 tile 的访问连续性。
3. 对尾部基本 tile 解析 M/N split，并过滤超出实际尺寸的 split。
4. 将有效普通 tile 和尾块 split 压缩成连续的 `tileIdx`，供 AIC/AIV 共同消费。

## 使用示例

```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>;

Scheduler::Params schedulerParams{baseM, baseN, 1U, 1U, 1U, 1U, 0U, 0U};
Scheduler scheduler(asc::te::make_shape(m, n, k), schedulerParams);
for (uint64_t tileIdx = 0; tileIdx < scheduler.GetTileCount(); ++tileIdx) {
    auto coord = scheduler.GetBlockCoord(tileIdx);
    auto shape = scheduler.GetBlockShape(coord);
    // 使用 coord 和 shape 构造当前 GM Tensor slice。
}
```

## 适用场景

- `GemmUniversal` Weight Prologue 特化的 AIC/AIV 共享 tile 调度。
- 需要 M/N SWAT 扫描和尾块 split 的单 Batch 矩阵乘。
- 不适用于携带 Batch、Grouped Matmul 或 StreamK 专用语义的 Scheduler 场景。
