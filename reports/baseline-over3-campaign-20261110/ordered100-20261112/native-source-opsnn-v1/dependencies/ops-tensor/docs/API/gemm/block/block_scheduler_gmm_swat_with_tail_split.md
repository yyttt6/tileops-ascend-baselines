# Block Scheduler GMM SWAT With Tail Split
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h)

## 功能说明
Grouped Matmul 的 BlockScheduler 组件，用于 QGMM MX Tensor API kernel。调度器按 group 逐次更新问题规模，在 group 间延续物理核分配位置以均衡负载，并在末组计算量较小时利用空闲核拆分 M/N tail block。
普通 QGMM 使用通用 block 坐标接口；GMMAQ ActivationQuant 使用 MX group/block 偏移接口；
SwiGLU MX 融合 Kernel 使用专用接口解析完整 N 的左右半区及量化输出偏移。

**框架参考**：[Block Scheduler 公共框架](./block_scheduler.md)

## 特殊约束

### 适用路径
适用于 grouped matmul 场景，当前由 `GemmUniversal` 的 QGMM MX 特化使用，不处理 split-K；K 维仅作为 shape 字段保留。

### 扫描策略
使用 SWAT 扫描策略，沿 M 轴以 `GMM_WINDOW_LEN` 为窗口组织 block，并在奇数行反向扫描 N 轴以提升局部性。

### Tail Split
Tail split 由调用方在末组按需触发。`SetTailAlign` 配置 M/N tail 的最小对齐粒度，`UpdateTailTile` 根据剩余空闲核数拆分末尾 M/N tail tile。

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| `GMM_WINDOW_LEN` | SWAT 扫描的 M 轴窗口长度，当前为 4 |
| `INNER_AXIS_MIN_SPLIT_VAL` | 内轴最小切分值，当前为 128 |

## 特殊类型别名

| 别名 | 含义 |
|------|------|
| `ProblemShape` | 问题总规模，`Shape<int64_t, int64_t, int64_t, int64_t>` |
| `BlockShape` | 单核基本块大小与 M/N tail split 偏移 |
| `BlockCoord` | block 坐标，`Coord<int64_t, int64_t, int64_t, int64_t>` |

## 特殊数据结构

### Params
```cpp
struct Params {
    int32_t baseM;
    int32_t baseN;
};
```

| 参数 | 说明 |
|------|------|
| `baseM` | M 轴基础 tile 大小 |
| `baseN` | N 轴基础 tile 大小 |

### SwiGLU 数据结构

```cpp
struct SwigluGroupParams {
    uint32_t groupIdx;
    int64_t groupMEndOffset;
    ProblemShape problemShape;
    bool singleW;
};

struct SwigluGroupInfo {
    int64_t aOffset;
    int64_t bOffset;
    int64_t scaleAOffset;
    int64_t scaleBOffset;
    int64_t outputOffset;
    int64_t outputScaleOffset;
    int64_t inputScaleK;
};

struct SwigluBlockInfo {
    BlockShape blockShape;
    BlockShape epilogueBlockShape;
    int64_t aMOffset;
    int64_t bLeftNOffset;
    int64_t bRightNOffset;
    SwigluOutputOffsets outputOffsets;
};
```

- `problemShape` 中的 N 是 SwiGLU 拆分前完整 N；调度本身使用的 N 是 `N / 2`。
- `groupMEndOffset` 是当前 group 在 M 轴上的累计结束位置，起始前缀为
  `groupMEndOffset - problemM`。
- `singleW=true` 时 B/ScaleB 基址按 `groupIdx` 在单个连续 Tensor 内偏移；为 `false` 时调用方已经
  选择当前 group 的 Tensor，group 基址偏移为 0。
- `bLeftNOffset` 与 `bRightNOffset` 分别定位完整 N 的左右半区，二者相差 `N / 2`。
- `epilogueBlockShape` 的 N 是单半区宽度；Y/YScale offset 均按 `N / 2` 输出空间计算。

## 特殊成员方法

### 构造函数
```cpp
__aicore__ inline BlockSchedulerGmmSwatWithTailSplit(const Params& params)
__aicore__ inline BlockSchedulerGmmSwatWithTailSplit(int32_t baseM, int32_t baseN, int32_t baseK)
```

功能：
- 初始化 M/N 轴基础 tile 大小。
- 三参数构造函数保留 `baseK` 以兼容 kernel 侧统一构造形式，调度器内部不使用 K 轴切分。

### UpdateNextProblem 函数
```cpp
__aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
```

功能：
- 刷新当前 group 的 `m/n/k` 问题规模。
- 重新计算 M/N tile 数、tail tile 大小、轮次和起止物理核。
- 从上一 group 的结束核后继续分配，实现 group 间负载均衡。

### UpdateBaseM 函数
```cpp
__aicore__ inline void UpdateBaseM(uint32_t baseM)
```

功能：
- 更新 M 轴基础 tile 大小，配合 kernel 侧 M 轴均衡逻辑使用。

### SetTailAlign 函数
```cpp
__aicore__ inline void SetTailAlign(uint32_t mTailAlign, uint32_t nTailAlign)
```

功能：
- 配置 M/N tail split 的对齐粒度。
- `UpdateTailTile` 会结合该粒度与剩余空闲核数计算 tail 拆分数。

### UpdateTailTile 函数
```cpp
__aicore__ inline void UpdateTailTile()
__aicore__ inline void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
```

功能：
- 无参版本根据当前 group 的 tail 大小和空闲核数自动计算 M/N tail split。
- 有参版本按调用方指定的 `mTailCnt/nTailCnt` 更新 tail split 状态。

### GetNextBlockCoord 函数
```cpp
__aicore__ inline bool GetNextBlockCoord(BlockCoord& blockCoord)
```

功能：
- 返回当前核本轮需要处理的 block 坐标。
- 当当前核无更多 block 时返回 `false`。

### GetBlockShape 函数
```cpp
__aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord)
```

功能：
- 根据 block 坐标返回当前单核 block 的实际 M/N 形状。
- 在 tail split 场景下，返回值的第 3/4 维携带 M/N split 偏移；当拆分后当前核没有有效工作量时返回 `{0, 0, 0, 0}`。

### GetEndBlockIdx 函数
```cpp
__aicore__ inline int64_t GetEndBlockIdx() const
```

功能：
- 返回当前 group 分配后的结束物理核索引。
- kernel 可据此判断末组是否有空闲核可用于 tail split。

### GMMAQ MX 地址接口

`UpdateMxGroup`、`GetMxGroupInfo` 和 `GetNextMxBlock` 仅服务于
`kernel_qgmm_mx_activation_quant.h`。它们不改变通用 SWAT 分核规则，而是在通用
Block 坐标之上补充 MX 数据布局所需的地址信息：

- Group 级的 A/B/ScaleA/ScaleB/Bias/Y/YScale 起始偏移。
- FP4 打包数据的元素到字节偏移换算。
- 当前 Block 的 M/N 坐标以及 YScale 的 64 元素分组偏移。

普通 QGMM 不调用这些接口，仍使用 `GetNextBlockCoord` 和 `GetBlockShape`。

### SwiGLU MX 地址接口

```cpp
__aicore__ inline void UpdateSwigluGroup(const SwigluGroupParams& params)
__aicore__ inline const SwigluGroupInfo& GetSwigluGroupInfo() const
__aicore__ inline bool GetNextSwigluBlock(SwigluBlockInfo& blockInfo)
```

`UpdateSwigluGroup` 根据 group 的 M 前缀、完整 N/K 和 `singleW` 计算 A/B/ScaleA/ScaleB/Y/YScale
基址。`GetNextSwigluBlock` 在通用 SWAT block 上补充左右权重 N offset、A 的 M offset，以及
Epilogue 使用的输出 offset。YScale 每 64 个输出元素保存两个 E8M0 scale，因此其行内 offset
按 32 个输出元素递增 1 字节。

## 调度流程

```text
构造 scheduler
    -> SetTailAlign
    -> 每个 group 调用 UpdateNextProblem
    -> 末组按需调用 UpdateTailTile
    -> GetNextBlockCoord 获取 block 坐标
    -> GetBlockShape 获取 block 形状和 tail split 偏移

SwiGLU MX：
UpdateNextProblem(M, N / 2, K)
    -> UpdateSwigluGroup(M, N, K)
    -> GetNextSwigluBlock
    -> 左右 B/ScaleB slice + Y/YScale offset
```

## 适用场景
- QGMM MX Tensor API kernel。
- group 间 M/N 规模动态变化的 grouped matmul。
- 末组 tile 数少于可用核数，需要利用空闲核拆分 M/N tail 的场景。
- SwiGLU MX 完整 N 的左右半区配对调度与输出偏移计算。
