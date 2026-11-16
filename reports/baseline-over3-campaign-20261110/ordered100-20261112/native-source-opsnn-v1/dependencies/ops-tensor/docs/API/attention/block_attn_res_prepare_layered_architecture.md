# BlockAttnResPrepare 分层结构说明

## 1. 边界

本文描述 Phase 1 混合 AIC+AIV 模板在 ops-tensor 中的公共组件。算子原型、Host Tiling、外部 Kernel
入口和 TilingData 位于 ops-transformer。

- 数据输入输出和中间计算均为 FP32，`validBlocks` 为 UINT64；
- 问题形状统一为 `[S,N,D,T]`；
- Attention 模板使用仓内 `AttentionUniversal`，不继承或特化 `GemmUniversal`；
- 两个矩阵乘复用 GEMM 的基础 `BlockMmad`，由 tuple 组合，不新增算子专用 MMAD。

## 2. 依赖与职责

```text
ops-transformer kernel entry
    |  显式组装 problemShape 和四组组件 Params
    v
AttentionUniversal specialization
    |-- Kernel: runtime token 合并、Tensor 构造、阶段编排、跨核同步、逻辑核映射
    |-- Scheduler: block 分配、BlockInfo 生成、AIV 行切分
    |-- MMAD tuple: MM1(Q*V^T)、MM2(E*V)
    `-- Epilogue: V 归约、RMS、softmax、空输入
            |-- Epilogue::Tile: ReduceSquare/RmsSoftmax/InitializeEmptySoftmax
            `-- Gemm::Tile: FillUb
```

依赖保持单向：

```text
attention kernel -> attention block / epilogue block / epilogue tile / gemm block / gemm tile
gemm -X-> attention
```

## 3. 文件

| 层 | 文件 | 作用 |
| --- | --- | --- |
| Policy | `include/blaze/attention/policy/dispatch_policy.h` | 编译期选择 BlockAttnResPrepare 特化 |
| Universal | `include/blaze/attention/kernel/kernel_universal.h` | Attention 主模板 |
| Kernel | `include/blaze/attention/kernel/kernel_block_attn_res_prepare.h` | 顶层组件组合和流水编排 |
| Scheduler | `include/blaze/attention/block/block_scheduler_block_attn_res_prepare.h` | T/S block 调度 |
| Epilogue | `include/blaze/epilogue/block/block_epilogue_block_attn_res_prepare.h` | AIV 后处理 |
| Epilogue Tile | `include/blaze/epilogue/tile/{reduce_square,rms_softmax,initialize_empty_softmax}.h` | 后处理 Tensor 级向量原语和架构分发 |
| GEMM Tile | `include/blaze/gemm/tile/compute.h` | 计算原语聚合头，提供通用 UB 清零（`FillUb`）等原语 |
| UT | `tests/ut/op_kernel/block_attn_res_prepare/` | 参数契约、Scheduler 和数值验证 |

没有独立的 `kernel_block_attn_res_prepare_params.h`。每个组件在自己的头文件内定义 `Params`，顶层 Kernel
只做组合。

## 4. 参数所有权

```cpp
Kernel::Params {
    problemShape;       // [S,N,D,T]
    mm1Params;          // MM1 地址、L1/L0 切分、stage
    mm2Params;          // MM2 地址、L1/L0 切分、stage
    epilogueParams;     // 后处理地址、D/UB/workspace 参数、epsilon
    schedulerParams;    // work units、逻辑核数和 T/S 调度参数
}
```

ops-transformer 从 TilingData 显式填充这些结构。MMAD 参数不由 Tensor Kernel 根据 base block 再推导；
Scheduler 也不读取整份算子 TilingData。

## 5. Scheduler

Kernel 首先计算逻辑核：

```text
AIC: logicalCoreIndex = GetBlockIdx()
AIV: logicalCoreIndex = GetBlockIdx() / GetTaskRation()
```

Kernel 根据运行时 `validN` 和 `mm1NAlign` 决定是否扩大 Host 的 token group，然后把更新后的调度参数与
`logicalCoreIndex` 传给 Scheduler。Scheduler 只生成 BlockInfo，并把 block 连续均分到逻辑核。

```text
sBlockIdx = blockIdx % sTileNum
tBlockIdx = blockIdx / sTileNum
blockShape = [blockS, validN, totalD, blockT]
blockCoord = [sOffset, 0, 0, tOffset]
```

## 6. MMAD tuple

tuple 元素 0 为 MM1：

```text
Q[blockS,D] * VGrouped[D,blockT*validN]
    -> dot[blockS,blockT*validN]
```

Residual 使用 batched `DNExt` B Tensor，基础 `BlockMmad` 把各 token 的 B 数据沿 BL1 N 方向拼接。
该能力由 MM1 的 `NON_CONTIGUOUS_TYPE_BATCHED_B` 策略显式开启；默认 `BlockMmad` 和 MM2 不进入该分支。

tuple 元素 1 为 MM2：

```text
E[blockS,validN] * V[validN,validD]
    -> numerator[blockS,validD]
```

MM2 对每个 token、每个 D tile 执行，Fixpipe 直接写最终 GM。

## 7. AIV Epilogue

对每个 token：

```text
sumSquare[n] = sum_d V[n,d]^2
rms[n]       = sqrt(sumSquare[n] / D + epsilon)
z[s,n]       = dot[s,n] / rms[n]
max[s]       = max_n z[s,n]
E[s,n]       = exp(z[s,n] - max[s])
sum[s]       = sum_n E[s,n]
```

Epilogue 的公开阶段函数都接收 Tensor。UB offset、长度和 stride 只在组件内部由 Tensor Layout 和 Params
构造，不暴露给调用者。

## 8. Workspace 与同步

Workspace 逻辑布局：

```text
per logical core:
| dot workspace | E slot 0 | E slot 1 |
```

Kernel 先创建 `[usedCoreNum,workspacePerCoreElems]` GM Tensor，再按逻辑核 `Slice()`。同组 AIC/AIV 使用
同一 slice。

跨核同步有三类：

- `dot ready`：MM1 完成后 AIC 通知两个 AIV；
- `E ready`：两个 AIV 都写完自己的 S 行后，AIC 才执行 MM2；
- `E buffer free`：MM2 消费完成后，AIC 释放对应 ping/pong slot。

## 9. 空输入

`validBlocks == 0` 时不执行 MM1/MM2。AIV0 遍历本逻辑核的输出 block，把 `numerator`、`logitMax` 和
`expSum` 全部写 0；兄弟 AIV 空闲。纯 AIV 模板和混合模板使用相同契约。
