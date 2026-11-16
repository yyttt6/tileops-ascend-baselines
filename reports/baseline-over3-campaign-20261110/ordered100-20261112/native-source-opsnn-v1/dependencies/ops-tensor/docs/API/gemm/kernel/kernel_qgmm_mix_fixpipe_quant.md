# Kernel QGMM MIX Fixpipe Quant

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mix_fixpipe_quant.h)

## 功能说明

`KernelGmmFixpipeQuant` 是 Grouped Matmul 的 Fixpipe 量化 Kernel。该组件面向算子侧已经准备好的 `int8_t` Cube 输入，组装 `BlockSchedulerGmmSwatWithTailSplit`、`BlockMmadA8W8FixpipeQuant` 和 `BlockEpiloguePerTokenScale`，支持 S8S4 与 S4S4 的公共矩阵乘及后处理流程。

本组件新增以下能力：

- 支持 Grouped Matmul 的多专家计算，并支持 `groupListType` 0、1、2。
- 支持 per-channel 和 per-group 两种反量化模式。
- 复用 Fixpipe 完成每个输出块的 scale 反量化。
- 支持可选的非对称量化 offset 修正和 per-token scale 后处理。
- 对最后一个专家复用 GMM tail-split 调度，提高尾块的核利用率。

INT4 数据展开、A4W4 激活预处理和行归约不属于本组件，由算子实现侧在调用 Tensor API 前完成。

## 组件关系

```text
KernelGmmFixpipeQuant
    -> BlockSchedulerGmmSwatWithTailSplit
    -> BlockMmadA8W8FixpipeQuant
    -> BlockEpiloguePerTokenScale
```

## 调度策略

该 Kernel 由 `KernelGroupedMmadWithScaleFixpipeQuant` 调度标签选择。`BlockMmad` 必须使用相同的 `ScheduleType`，防止普通 QBMM、StreamK 和 GMM Fixpipe 组件发生错误组合。

Kernel 逐专家读取 M 轴分组信息，为当前专家更新问题规模，并按 Scheduler 返回的基本块坐标调用 BlockMmad。per-channel 与 per-group 的选择由 `GMMTiling::quantMode` 决定。

## 量化模式

### Per-channel

每个专家对应一个长度为 N 的 scale 向量。BlockMmad 完成完整 K 轴累加后，通过 Fixpipe 应用当前 N 分片的 scale。

### Per-group

K 轴按照 `quantGroupSize` 切分，每个 K-group 对应一个长度为 N 的 scale 向量。Kernel 将 group 数量和 group 大小交给 BlockMmad，由 BlockMmad 完成各 K-group 的计算与结果累加。

## 主要参数

`Params` 由三部分组成：

| 参数 | 说明 |
|------|------|
| `mmadParams` | A、B、scale 及中间输出相关的 BlockMmad 参数 |
| `epilogueParams` | per-token scale、offset、row sum 和最终输出相关的后处理参数 |
| `groupListGmAddr` | 专家分组信息地址 |
| `gmmParams` | GMM 问题规模、基本块、量化模式及 group 配置 |

`GMMTiling` 中与本功能直接相关的字段包括 `groupNum`、`m/n/k`、`baseM/baseN/baseK`、`quantMode`、`quantGroupSize` 和 `groupListType`。

## 使用约束

- Kernel 编译期通过 `static_assert` 校验模板组合。
- `AType` / `BType` 仅支持 `int8_t`。
- Fixpipe scale 类型仅支持 `uint64_t`。
- `CType` 仅支持 `half`，`BiasType` 仅支持 `int32_t`。
- `LayoutA` 仅支持 `nd_ext_layout_ptn`。
- `LayoutB` 仅支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- `LayoutC` / `LayoutBias` 支持 ND 类布局，不支持 `nz_layout_ptn`、`zn_layout_ptn`。
- `BlockEpilogue::FixpipeType` 需与 `CType` 一致。
- `quantMode` 仅使用 `PERCHANNEL_MODE` 或 `PERGROUP_MODE`。
- per-group 模式下，算子侧应保证 `quantGroupSize` 和 scale 的 group 维度与 K 轴切分一致。
- 当前 GMM 组合不在 Cube MMAD 中处理 API bias。
