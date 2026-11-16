# BlockEpilogueFinalizeRouting

## 功能说明

`BlockEpilogueFinalizeRouting` 是 `GroupedMatmulFinalizeRouting` TensorAPI Kernel 的通用 AIV 侧后处理组件。它接收 MMAD 产生的 FP32 结果，融合逐 token 的 logit，并根据 `rowIndex` 对 y 进行原子加写回。

一个 AIC tile 会由两个 AIV 子核在 M 维均分处理。L0C-to-UB 使用单块 FP32 缓冲，输出 UB 和 logit UB 使用 ping-pong 缓冲，使向量计算与 GM 搬运能够交替执行。

## 模板约束

| 类型 | 约束 |
| --- | --- |
| `DataTypeIn` | FP32 MMAD 输出 |
| `DataTypeOut` | FP32 或 BF16 |
| `DataTypeLogit` | FP32 或 BF16；FP32 输出时 BF16 logit 在组件内转换为 FP32 |
| `RowIndexType` | FP32 输出时为 INT64；BF16 输出时为 INT32 |
| FP32 输出 | logit 以 FP32 参与乘法 |
| BF16 输出 | logit 可为 FP32 或 BF16 |

## 参数

| 字段 | 说明 |
| --- | --- |
| `yGmAddr` | 输出 y 的 GM 地址 |
| `logitGmAddr` | logit 的 GM 地址 |
| `rowIndexGmAddr` | rowIndex 的 GM 地址 |
| `x1ScaleGmAddr`、`x2ScaleGmAddr`、`biasGmAddr` | 保持与 Kernel 参数结构一致的地址字段；后处理不读取这些字段 |
| `baseM`、`baseN` | tile 基础大小 |

`UpdateGlobalAddr` 在处理新 group 时根据 group 的基地址更新 logit 和 rowIndex 的 GM 视图。

## BF16 输出路径

BF16 输出时，Epilogue 保持与量化通路一致的 BF16 计算顺序：

1. 将 L0C-to-UB 中的 FP32 MMAD 结果转换为 BF16。
2. logit 为 FP32 时先转换为 BF16；logit 已为 BF16 时直接使用。
3. 使用 BF16 乘法得到输出，并写入 BF16 输出 UB。
4. 使用 BF16 原子加写回 `y[rowIndex, nOffset:nOffset + curN]`。

FP32 输出路径保持 FP32 logit 和 FP32 MMAD 结果的乘法，再以 FP32 原子加写回。
当 logit 输入为 BF16 时，组件会先将其转换为 FP32，再执行相同的 FP32 乘法路径。算子接口的运行时 dtype 校验仍以接口约束为准。

## 执行流程

1. 根据 AIV 子核编号切分当前 tile 的 M 维，获取本子核处理范围。
2. 从 L0C-to-UB 缓冲读取 FP32 结果，并按行从 GM 读取 logit。
3. 向量侧执行结果与 logit 的广播乘法及所需类型转换。
4. 对本子核的每一行读取 `rowIndex`，通过 `SetAtomicAdd` 将结果写入 y 的对应行和 N 范围。
5. 通过 V/MTE2/MTE3 event 管理 logit 搬入、向量计算和 y 写回的双缓冲依赖。

## 注意事项

- 写回使用原子加，目标 y 必须满足调用方约定的初始化条件。
- `rowIndex` 的类型受输出 dtype 约束；不能混用 BF16 输出与 INT64 rowIndex，或 FP32 输出与 INT32 rowIndex。
- L0C-to-UB 缓冲以 FP32 保存 MMAD 结果，BF16 转换发生在向量后处理阶段。

## 相关文件

- 实现：[block_epilogue_finalize_routing.h](../../../../include/blaze/epilogue/block/block_epilogue_finalize_routing.h)
- Kernel：[kernel_qgmm_mx_mix_finalize_routing.h](../../gemm/kernel/kernel_qgmm_mx_mix_finalize_routing.md)
