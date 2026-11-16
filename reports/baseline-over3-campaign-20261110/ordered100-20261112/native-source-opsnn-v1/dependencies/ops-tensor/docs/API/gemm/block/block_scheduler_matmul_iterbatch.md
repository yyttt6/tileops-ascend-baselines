# Block Scheduler IterBatch
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_iterbatch.h)

## 功能说明
IterBatch 调度器，支持 batch 分组。将 batch 维按 `iterBatchL1` 分组生成 tile 序列，尾组余数自动分配。适用于 GemmUniversal iterbatch 特化的批量矩阵乘场景。

## 模板参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| ProblemShape_ | Shape<int64_t, int64_t, int64_t, int64_t> | - | 问题规模 `(m, n, k, batch)` |

## Params 参数结构

| 字段 | 类型 | 说明 |
|------|------|------|
| baseM/baseN/baseK | uint32_t | M/N/K 方向 tile 大小（BlockMmad/kernel 消费） |
| iterBatchL1 | uint32_t | L1 同时驻留的 batch 数（本调度器消费，决定 tile 分组粒度） |
| iterBatchL0 | uint32_t | L0 流水线并行 batch 数（BlockMmad/kernel 消费） |
| isHf32 | uint8_t | FP32 HF32 模式（kernel `SetHF32/UnsetHF32` 消费） |
| l2CacheDisable | uint32_t | L2 cache 模式（kernel `SetL2Cache` 消费，默认 `L2_CACHE_DEFAULT`） |

## 接口

| 接口 | 说明 |
|------|------|
| `GetBlockNums()` | 总 block 数（batch 分组数，`CeilDiv(b, iterBatchL1)`） |
| `GetCoreNums(blockNum)` | 实际参与核数（`min(GetBlockNums(), blockNum)`） |
| `GetBlockShape(blockIdx)` | 该 block 的形状 `(m, n, k, batch 数)`，尾组 batch 为余数 `b - blockIdx * iterBatchL1` |
| `GetBlockCoord(blockIdx)` | 该 block 的起始坐标 `(0, 0, 0, blockIdx * iterBatchL1)` |
