# BlockSchedulerWqmmTailResplit

> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_wqmm.h)

## 功能说明

该调度器将矩阵乘的 M/N 区域分配给 AIC 及其对应的两个 AIV。
Host 提供 N 轴主块和两段尾块的数量、长度，调度器据此计算各核的分块坐标和有效长度。
分块指一次处理的矩阵子区域；尾块是轴末端不足主块大小的区域，可进一步分为两段以分配给不同核。
K 轴遍历由 Kernel 和 BlockMmad 负责。

## 模板与参数

模板参数 `ProblemShape_` 表达 `(M,N,K)`，分别对应输入 `x` 的行数、权重 `weight` 的输出通道数及归约长度。

| `Params` 成员 | 类型 | 含义 |
| --- | --- | --- |
| `mL1Tile` | `int64_t` | 每次处理的最大 M 长度，单位为元素 |
| `mainBlockCount`、`mainBlockSize` | `uint64_t` | N0 主块数量、每块 N 长度 |
| `firstTailBlockCount`、`firstTailBlockSize` | `uint64_t` | N1 尾块数量、每块 N 长度 |
| `secondTailBlockCount`、`secondTailBlockSize` | `uint64_t` | N2 尾块数量、每块 N 长度 |
| `cubeNumBlocksM`、`cubeNumBlocksN` | `uint64_t` | M/N 方向的 AIC 分核数 |

各 N 长度以元素计。调用方须保证 M/N、`mL1Tile` 和分核数为正，
三个 N 区间的 `count*size` 之和等于 N，非空区间的分块长度为正。
主块和尾块的配置均须满足所用 Kernel 的片上缓冲区容量要求。
`Params` 成员默认值均为 0，使用前需要设置合法的分块配置。

## 公开接口

| 方法 | 含义及返回值 |
| --- | --- |
| 构造函数 `(const ProblemShape_& problemShape, const Params& params)` | 根据当前核号初始化负责的区间 |
| `GetCoreNums()` | AIC 数量，即 `cubeNumBlocksM*cubeNumBlocksN` |
| `GetBlockNums()` | 当前核的 `(M,N0,N1,N2)` 分块数量；可用 `auto` 接收 |
| `GetBlockCoordM(mIdx)` | M 起始坐标，类型为 `uint64_t` |
| `GetBlockShapeM(coordM)` | M 有效长度，类型为 `uint64_t` |
| `GetBlockCoordN<N_RANGE>(nIdx)` | N 起始坐标，类型为 `uint64_t` |
| `GetBlockShapeN<N_RANGE>(coordN)` | N 有效长度，类型为 `uint64_t` |

坐标和长度均以元素计。`N_RANGE=0/1/2` 分别表示 N0、N1、N2。
坐标接口接收当前核内的分块序号；长度接口接收该轴的标量坐标，返回值裁剪到对应区间的边界。
Kernel 根据这些标量构造 Tensor 切片所需的 Shape/Coord。

## 分配规则

M 轴先按 `ceil(M/cubeNumBlocksM)` 划分核间区间，每个核再以 `mL1Tile` 为步长遍历自己的区间。
N0、N1、N2 在 N 轴上顺次相接：N0 起点为 0，后一区间起点为前一区间终点。

N0 和 N1 都从 N 方向的第 0 个核开始轮转分配；N2 从 N1 最后分配位置的下一核接续。
设当前 N 方向核号为 `nDimIdx`，其第一个 N2 分块序号为：

```text
(nDimIdx + cubeNumBlocksN - firstTailBlockCount % cubeNumBlocksN) % cubeNumBlocksN
```

AIV 将自身核号除以子核数量，得到对应 AIC 的逻辑核号。
AIC 与 AIV 按 M → N0 → N1 → N2 的相同顺序遍历；每个 M 分块均遍历其负责的全部 N 区间。

## 分配示例

取 `M=65`、`N=159`，`cubeNumBlocksM=1`、`cubeNumBlocksN=2`、`mL1Tile=32`，
N0 为 2 块、每块 64；N1 为 1 块、长度 15；N2 为 1 块、长度 16。
两个核的 N 分配如下，区间左闭右开：

| 区间 | N 方向核 0 | N 方向核 1 |
| --- | --- | --- |
| N0 `[0,128)` | `[0,64)` | `[64,128)` |
| N1 `[128,143)` | `[128,143)` | 无 |
| N2 `[143,159)` | 无 | `[143,159)` |

两个核都处理 M 起点 0、32、64，对应有效长度 32、32、1。
每个 M 分块内，核 0 依次处理 N0 和 N1，核 1 依次处理 N0 和 N2。
