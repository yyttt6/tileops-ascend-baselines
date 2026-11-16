# Kernel Qbmm Cube Without Batch

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_cube_without_batch.h)

## 功能说明

该 Kernel 面向单 Batch 量化矩阵乘场景，仅在 AIC 上执行。Kernel 由
`BlockMmadA8W8FixpipeQuant` 与 `BlockSchedulerQuantBatchMatmulV3` 组装，并使用 Fixpipe
搬出结果、按需执行 scale 反量化。由于 Batch 固定为 1，调度过程不涉及多 Batch 广播、
Batch 地址换算和 Batch 循环。

支持的数据类型、Layout、scale 模式、Bias、Atomic Add 和 L1/L0 约束与
[Kernel Qbmm Cube](./kernel_qbmm_cube.md) 一致。

## 调度约束

- `BlockMmad::DispatchPolicy::ScheduleType` 必须为
  `KernelMmadWithScaleFixpipeQuantWithoutBatch`。
- `problemShape` 的 Batch 固定为 1。
- AIV 不参与计算，Kernel entry 在 AIV 上启动后直接返回。
- `nBufferNum` 支持 2、3 或 4。非全载三缓冲模式下，A/B 数据各使用三个 L1 buffer；
  A 全载三缓冲模式下，A 常驻 L1，B 使用三个 L1 buffer。per-tensor 标量 scale 不占用 L1；
  per-channel scale 使用两个 L1 buffer，每个 buffer 可容纳 `baseN` 个 scale；启用 Bias 时，
  Bias 同样使用两个 L1 buffer。

## 参数结构

```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockSchedulerParams schParams;
    QBMMTiling qbmmParams;
};
```

`QBMMTiling` 包含 `x1QuantMode/x2QuantMode`、`kAL1/kBL1`、`nBufferNum`、
`baseM/baseN/baseK`、`isBias`、`dbL0C` 和 `bMustHitL2`，不包含 Batch 广播字段。

## 组装示例

```cpp
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
    FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, AscendC::Std::tuple<BType, ScaleType>, LayoutB,
    CType, LayoutC, BiasType, LayoutC>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```
