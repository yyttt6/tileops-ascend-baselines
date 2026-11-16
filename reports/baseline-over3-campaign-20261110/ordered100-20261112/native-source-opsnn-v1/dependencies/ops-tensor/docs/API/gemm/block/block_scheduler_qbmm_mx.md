# Block Scheduler Quant Batch Matmul
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_qbmm.h)

## 功能说明
量化 Batch Matmul 调度器，支持多 Batch 维度切分、尾块切分、负载均衡、Z 型扫描。适用于 Quant Batch Matmul MX Kernel 场景，支持 MxFP4/MxFP8 量化数据类型。

**继承自**：[Block Scheduler 公共框架](./block_scheduler.md)

## 模板参数

### 模板参数列表
```cpp
template <
    typename ProblemShape_,     // 问题规模类型
    uint64_t FullLoadMode_,     // 全载模式
    typename LayoutA_,          // A 矩阵布局类型
    typename LayoutB_,          // B 矩阵布局类型
    typename AType_>            // A 矩阵数据类型
class BlockSchedulerQuantBatchMatmulV3;
```

### 参数详解

| 参数 | 类型约束 | 默认值 | 说明 |
|------|----------|--------|------|
| ProblemShape_ | `Shape<int64_t, int64_t, int64_t>` | - | 问题规模 `(m, n, k)` |
| FullLoadMode_ | `uint64_t` | 0 | 全载模式：0=非全载，1=A全载 |
| LayoutA_ | Layout Pattern 类型 | - | A 矩阵布局类型，用于判断转置 |
| LayoutB_ | Layout Pattern 类型 | - | B 矩阵布局类型，用于判断转置和 NZ 格式 |
| AType_ | `fp4x2_e2m1_t`, `fp4x2_e1m2_t`, `fp8_e5m2_t`, `fp8_e4m3fn_t` | - | A 矩阵数据类型，用于判断 NZ 对齐粒度 |

**转置判断**：
```cpp
// 通过 IsTrans trait 自动判断转置
bool TRANS_A = IsTrans<LayoutA>::value;  // A 矩阵是否转置
bool TRANS_B = IsTrans<LayoutB>::value;  // B 矩阵是否转置
```

**NZ 格式判断**：
```cpp
// 通过 IsWeightNz trait 判断 B 矩阵是否为 NZ 格式
bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
```

### AType_ 参数详解

**支持的量化数据类型**：
| 数据类型 | 说明 | NZ 对齐粒度（!TRANS_B） |
|---------|------|----------------------|
| `fp4x2_e2m1_t` | MxFP4 E2M1 格式 | 64（Align64） |
| `fp4x2_e1m2_t` | MxFP4 E1M2 格式 | 64（Align64） |
| `fp8_e5m2_t` | MxFP8 E5M2 格式 | 32（Align32） |
| `fp8_e4m3fn_t` | MxFP8 E4M3FN 格式 | 32（Align32） |

**NZ 对齐粒度说明**：
- `WEIGHT_NZ = true` 且 `!TRANS_B` 时，N 轴按 AType 选择对齐粒度
  - **FP4 类型**：调用 `Align64`（对齐到 64）
  - **FP8 类型**：调用 `Align32`（对齐到 32）
- `TRANS_B = true` 时，N 轴按 `BLOCK_CUBE` 对齐（当前实现调用 `Align16`）

### 转置判断
根据布局类型判断转置：
- **TRANS_A**：`IsTrans<LayoutA_>::value`
- **TRANS_B**：`IsTrans<LayoutB_>::value`

### 尾块切分
支持尾块 tile 切分：
- **mTailTile**：M 轴尾块切分数量
- **nTailTile**：N 轴尾块切分数量
- **totalTailTile**：总尾块切分数量（mTailTile × nTailTile）

### Z 型扫描
使用 Z 型扫描策略：
- **WINDOW_LEN = 4**：扫描窗口大小
- **正向扫描**：偶数行（rowIdx % 2 == 0）
- **反向扫描**：奇数行（rowIdx & 1）

### 负载均衡
支持负载均衡配置：
- **mBaseNormCnt**：M 轴正常 tile 数量
- **mBaseTailMain**：M 轴尾块主尺寸
- **nBaseNormCnt**：N 轴正常 tile 数量
- **nBaseTailMain**：N 轴尾块主尺寸

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| WINDOW_LEN | Z 型扫描窗口长度（4） |
| TRANS_A | A 矩阵是否转置 |
| TRANS_B | B 矩阵是否转置 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BlockShape | Block 形状：`Shape<int64_t, int64_t, int64_t, int64_t>` |
| BlockCoord | Block 坐标：`Coord<int64_t, int64_t>`，仅包含 `(mTileIdx, nTileIdx)` |
| ProblemShape | 问题规模类型（模板参数） |
| AType | A 矩阵数据类型（模板参数） |

## 特殊数据结构

### Params
```
struct Params {
    int64_t baseM;              // L0 M 维度 base 大小
    int64_t baseN;              // L0 N 维度 base 大小
    int64_t mTailTile;          // M 轴尾块切分数量
    int64_t nTailTile;          // N 轴尾块切分数量
    int64_t mBaseTailSplitCnt;  // M 轴尾块 L1 切分数量
    int64_t nBaseTailSplitCnt;  // N 轴尾块 L1 切分数量
    int64_t mTailMain;          // M 轴尾块主尺寸
    int64_t nTailMain;          // N 轴尾块主尺寸
};
```

## 特殊成员变量

| 变量 | 说明 |
|------|------|
| baseM_, baseN_ | L0 base 形状 |
| mBlockNums_, nBlockNums_, blockNums_ | M/N 轴及总 block 数量 |
| mBaseNormCnt_, nBaseNormCnt_ | 正常 block 数量 |
| mBaseTailMain_, nBaseTailMain_ | 尾块主尺寸 |
| mBaseTailLast_, nBaseTailLast_ | 尾块最后尺寸 |
| mainWindow_, tailWindow_ | M 轴主扫描窗口、尾扫描窗口 |
| blockIdx_, coreNums_ | 当前 Block 索引、总核数 |
| startBlockIdx_, endBlockIdx_ | 起始/结束 Block 索引 |
| roundIdx_, roundNums_ | 当前轮次、总轮次 |
| mTailTile_, nTailTile_, totalTailTile_ | 尾块切分数量 |
| mainRow_ | 主行数 |

尾块切分偏移由 Scheduler 成员保存；`GetBlockShape` 计算当前 block 形状并刷新偏移，随后调用 `GetTileCoord` 获取包含尾块切分偏移的 GM 坐标。

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockSchedulerQuantBatchMatmulV3(const ProblemShape& shape, const Params& params)
```
功能：初始化 BlockSchedulerQuantBatchMatmulV3，计算 block 切分、尾块参数、轮次等。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k)` |
| params | Params | 调度参数 |

执行流程：
1. 读取问题规模中的 `m`, `n`，设置 `baseM_`, `baseN_`
2. 计算 block 数量：`mBlockNums_`, `nBlockNums_`, `blockNums_`
3. 计算扫描窗口：`mainWindow_`, `mainRow_`, `tailWindow_`
4. 计算轮次：`endBlockIdx_`, `roundNums_`
5. 计算尾块参数：根据 `TRANS_A` 和 `TRANS_B` 计算尾块尺寸

### UpdateTailTile
```
__aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
```
功能：更新尾块切分数量，重新计算结束 Block 索引和轮次。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| mTailTile | uint32_t | M 轴尾块切分数量 |
| nTailTile | uint32_t | N 轴尾块切分数量 |

### GetTotalCnt
```
__aicore__ inline int64_t GetTotalCnt()
```
功能：返回总 block 数量（`blockNums_`）。为保持现有接口兼容性，函数名仍为 `GetTotalCnt`。

### GetEndBlockIdx
```
__aicore__ inline int64_t GetEndBlockIdx()
```
功能：返回结束 Block 索引（`endBlockIdx_`）。


### GetBlockShape
```
template <QuantMode AQuantMode_, QuantMode BQuantMode_, bool WeightNz_ = false>
__aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
```
功能：返回当前 Block 的形状，支持量化模式和 NZ 格式。

说明：`CalSingleCoreShapeByCoord` 为私有辅助方法，由 `GetBlockShape` 内部调用，不属于对外接口。

参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| AQuantMode_ | QuantMode | A 矩阵量化模式（PERGROUP/PERBLOCK） |
| BQuantMode_ | QuantMode | B 矩阵量化模式（PERGROUP/PERBLOCK） |
| WeightNz_ | bool | B 矩阵是否为 NZ 格式 |
| blockCoord | BlockCoord | Block 坐标 |

返回值：`BlockShape {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset}`。
其中 `mSplitAddrOffset`、`nSplitAddrOffset` 为当前尾块切分的 GM 坐标增量，同时会记录在 Scheduler 内部供 `GetTileCoord` 使用。

特殊逻辑：
- **尾块切分判断**：`totalTailTile_ > 1 && roundIdx_ == roundNums_`
- **FP4 对齐**：FP4 + `TRANS_A` 时 M 对齐到 2，FP4 + `!TRANS_B` 时 N 对齐到 2
- **PERBLOCK 模式**：对齐到 PER_BLOCK_SIZE 或 2 的幂次方
- **NZ 格式对齐**：`!TRANS_B` 时根据 AType 对齐到 64/32，`TRANS_B` 时对齐到 BLOCK_CUBE（16）

### UpdateNextBatchBlockRoundParams
```
__aicore__ inline void UpdateNextBatchBlockRoundParams()
```
功能：更新下一 Batch 的 Block 轮次参数。
执行流程：
1. 更新 `startBlockIdx_` 和 `endBlockIdx_`
2. 重置 `roundIdx_ = 0`
3. 重新计算 `roundNums_`

### GetTileIdx
```
__aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
```
功能：获取当前轮次的 block 坐标。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockCoord | BlockCoord& | Block 坐标（原地修改） |

返回值：
- **true**：当前轮次有效，返回 block 坐标
- **false**：当前轮次结束

执行流程：
1. 判断轮次结束：`roundIdx_ >= roundNums_`
2. 计算 block 索引：根据全载模式计算
3. Z 型扫描：计算 `blockCoordM`, `blockCoordN`
4. 反向扫描：奇数行反向（`blockCoordN = nBlockNums_ - 1 - blockCoordN`）
5. 更新 `roundIdx_++`

### GetTileCoord
```
__aicore__ inline void GetTileCoord(BlockCoord blockCoord, int64_t& mPos, int64_t& nPos)
```
功能：根据 Block 坐标和 Scheduler 内部记录的尾块切分偏移计算 GM 地址偏移。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockCoord | BlockCoord | Block 坐标 |
| mPos | int64_t& | M 轴 GM 偏移（原地修改） |
| nPos | int64_t& | N 轴 GM 偏移（原地修改） |

## 调用示例

### 组件组装
```
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
using LayoutA = asc::te::nz_layout_ptn;
using LayoutB = asc::te::nz_layout_ptn;
using AType = fp4x2_e2m1_t;
constexpr uint64_t FULL_LOAD_MODE = 0;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, FULL_LOAD_MODE, LayoutA, LayoutB, AType>;
```

### 参数准备
```
BlockScheduler::Params params = {
    baseM,              // L0 M 维度 base（如 128）
    baseN,              // L0 N 维度 base（如 128）
    mTailTile,          // M 轴尾块切分数量（如 1）
    nTailTile,          // N 轴尾块切分数量（如 1）
    mBaseTailSplitCnt,  // M 轴尾块 L1 切分数量（如 1）
    nBaseTailSplitCnt,  // N 轴尾块 L1 切分数量（如 1）
    mTailMain,          // M 轴尾块主尺寸（如 1）
    nTailMain           // N 轴尾块主尺寸（如 1）
};
```

### 组件初始化
```
ProblemShape shape{m, n, k};
BlockScheduler scheduler(shape, params);
```

### 更新尾块切分
```
scheduler.UpdateTailTile(mTailTile, nTailTile);
```

### 获取 block 数量
```
int64_t totalCnt = scheduler.GetTotalCnt();
```

### Block 循环处理
```
BlockCoord blockCoord{0, 0};
while (scheduler.GetTileIdx(blockCoord)) {
    // 获取 Block 形状
    constexpr auto A_QUANT_MODE = QuantMode::MX_PERGROUP_MODE;
    constexpr auto B_QUANT_MODE = QuantMode::MX_PERGROUP_MODE;
    constexpr bool WEIGHT_NZ = true;
    auto blockShape = scheduler.GetBlockShape<A_QUANT_MODE, B_QUANT_MODE, WEIGHT_NZ>(blockCoord);

    int64_t singleCoreM = Get<0>(blockShape);
    int64_t singleCoreN = Get<1>(blockShape);
    int64_t mSplitOffset = Get<2>(blockShape);
    int64_t nSplitOffset = Get<3>(blockShape);

    // 获取 GM 地址偏移
    int64_t mPos, nPos;
    scheduler.GetTileCoord(blockCoord, mPos, nPos);

    // 执行 BlockMmadMX 计算
    // ...
}
```

### 更新下一 Batch
```
scheduler.UpdateNextBatchBlockRoundParams();
```

## 数据流

### Tile 切分流程
```
问题规模 (m, n, k)
    ↓
L0 tile 切分 (baseM, baseN)
    ↓
tile 数量计算 (mCnt × nCnt)
    ↓
尾块参数计算 (mBaseNormCnt, mBaseTailMain, nBaseNormCnt, nBaseTailMain)
    ↓
轮次计算 (round, startBlockIdx, endBlockIdx)
    ↓
Z 型扫描 (mCoreNum, mainRow, mTailCoreNum)
    ↓
Block 形状/坐标 (singleCoreM, singleCoreN, mPos, nPos)
```

尾块切分时，`GetBlockShape` 会刷新 Scheduler 内部的切分偏移；调用方随后直接调用 `GetTileCoord(blockCoord, mPos, nPos)` 即可得到包含尾块切分偏移的 GM 坐标。

### 量化模式对齐流程
```
GetBlockShape<QuantMode, QuantMode, WEIGHT_NZ>
    ↓
尾块切分判断 (totalTailTile > 1 && roundIdx == round)
    ↓
FP4 对齐：TRANS_A → M 对齐到 2，!TRANS_B → N 对齐到 2
    ↓
PERBLOCK 模式：对齐到 PER_BLOCK_SIZE 或 2 的幂次方
    ↓
NZ 格式对齐：!TRANS_B → FP4 Align64 / FP8 Align32，TRANS_B → Align16
    ↓
返回 BlockShape {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset}
```

### Z 型扫描流程
```
blockIdx 计算
    ↓
rowIdx = blockIdx / nCnt / mCoreNum
    ↓
rowIdx < mainRow：blockCoordM = rowIdx × mCoreNum + blockIdx % mCoreNum
    ↓
rowIdx == mainRow：尾窗口计算
    ↓
rowIdx & 1：反向扫描（blockCoordN = nCnt - 1 - blockCoordN）
```

## 性能优化建议

### baseM/baseN 配置
- **建议值**：根据量化数据类型选择（如 128）
- **NZ 对齐**：`WEIGHT_NZ` 场景下，`!TRANS_B` 时 N 轴按 FP4 64 / FP8 32 对齐，`TRANS_B` 时按 16 对齐

### 尾块切分配置
- **mTailTile/nTailTile**：建议尾块切分数量不超过 4
- **mBaseTailSplitCnt/nBaseTailSplitCnt**：建议 L1 尾块切分数量为 1（不切分）

### 全载模式选择
- **非全载模式（FullLoadMode = 0）**：适用于一般场景
- **A 全载模式（FullLoadMode = A_FULL_LOAD_MODE）**：适用于大 K、小 M 场景

### 量化模式选择
- **PERGROUP_MODE**：per-group 量化，适用于小规模量化
- **PERBLOCK_MODE**：per-block 量化，对齐要求更高

### NZ 格式优化
- **WEIGHT_NZ = true**：B 矩阵 NZ 格式，提升 L1/L0 搬运效率
- 对齐要求：`!TRANS_B` 时根据 AType 选择 64/32 对齐，`TRANS_B` 时对齐到 16

### 适用场景
- **Quant Batch Matmul MX Kernel**：量化 Batch Matmul
- **MxFP4/MxFP8 量化**：支持 FP4 E2M1/E1M2 和 FP8 E5M2/E4M3FN
- **多 Batch 维度**：支持 4 维 Batch（batchA/A2/A3/A4）
- **负载均衡**：动态调整 tile 分配
