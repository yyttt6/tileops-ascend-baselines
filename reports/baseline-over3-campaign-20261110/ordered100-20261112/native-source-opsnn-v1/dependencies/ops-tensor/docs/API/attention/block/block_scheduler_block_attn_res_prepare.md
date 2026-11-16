# BlockAttnResPrepare BlockScheduler

> [代码位置](../../../../include/blaze/attention/block/block_scheduler_block_attn_res_prepare.h)

`BlockSchedulerBlockAttnResPrepare<ProblemShape>` 负责 Phase 1 AIC+AIV 模板的运行时任务切分。
`ProblemShape` 按 `[S,N,D,T]` 解释。

## 参数

```cpp
struct Params {
    uint32_t totalWorkUnits;
    uint32_t usedCoreNum;
    uint32_t baseT;
    uint32_t baseS;
    uint32_t sTileNum;
    uint32_t mm1NAlign;
};
```

构造函数接收 `(problemShape, params, validN, logicalCoreIndex)`。物理核到逻辑核的映射由 Kernel 完成，
Scheduler 不依赖 `GetBlockIdx()`，便于独立验证其调度策略。

## 调度策略

- Kernel 根据运行时 `validN` 计算 `baseT/totalWorkUnits`，Scheduler 不再修改切分参数；
- Scheduler 按连续区间把 block 近似均分给各逻辑核；
- Scheduler 根据 `baseT/baseS/sTileNum` 生成每个 `BlockInfo`；
- Scheduler 根据 `GetTaskRation()` 把一个 block 的 S 行近似均分给同组 AIV。

`GetNextBlock()` 返回：

```cpp
blockShape = [blockS, validN, totalD, blockT];
blockCoord = [sOffset, 0, 0, tOffset];
```

`GetAivRowRange()` 再把一个 block 的 S 行近似均分给同组 AIV。Scheduler 不负责 Tensor 构造、矩阵乘、
GM 搬运或跨核同步。
