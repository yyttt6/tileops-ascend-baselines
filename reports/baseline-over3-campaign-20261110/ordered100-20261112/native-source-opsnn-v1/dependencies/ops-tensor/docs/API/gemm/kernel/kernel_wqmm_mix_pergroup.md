# Kernel Wqmm Mix Pergroup
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_wqmm_mix_pergroup.h)

## 功能说明

本页介绍 `GemmUniversal` 的 arch35 T-CG per-group 权重反量化特化：AIV（Vector）侧
`KernelPergroupWeightPrologue` 把 packed FP4 权重乘 per-group scale 反量化成 FP8 写入
共享 L1；AIC（Cube）侧按相同的 Scheduler tile 取用这些数据，执行 FP8×FP8 MMAD，并在
Fixpipe 阶段乘
per-channel yScale 量化输出。计算公式：

```text
y = (x1 @ (x2 ⊙ x2Scale)) ⊙ yScale
```

**配套组件**：[Block Mmad Wqmm Mix Prologue Fixpipe Quant](../block/block_mmad_wqmm_mix_prologue_fixpipe_quant.md)、
[Block Scheduler Wqmm Block Split](../block/block_scheduler_wqmm_block_split.md)、
[Dequant](../../epilogue/tile/arch35/dequant.md) 和
[Copy UB To L1](../tile/copy_ub_to_l1.md)

## 特殊约束

### 架构支持

该特化仅在 `__NPU_ARCH__ == 3510`（arch35）下定义。

### 调度策略

`BlockMmad::DispatchPolicy::ScheduleType` 必须为 `KernelMixWithWeightPergroupPrologue`
（对应 `MatmulWithWeightQuantPergroup`）。`BlockEpilogue` 固定为 `void`：量化输出由
`BlockMmad` 内的 Fixpipe 直接完成。Scheduler 使用
`BlockSchedulerWqmmBlockSplit`（固定核分核 + 核内 swizzle），AIC/AIV 构造相同
tile 序列逐 tile 对齐。

### 数据类型和布局

| 数据 | 类型 | Layout | 说明 |
| :--- | :--- | :--- | :--- |
| x1（A） | `fp8_e4m3fn_t` | `nd_ext_layout_ptn` | 激活矩阵，非转置 |
| x2（B） | `fp4x2_e2m1_t` | NZ: `nz_layout_ptn` / ND: `dn_ext_layout_ptn` | packed FP4 转置权重 |
| x2Scale | `half` / `bfloat16_t` | ND: `dn_ext_layout_ptn` / NZ: `nd_ext_layout_ptn` | per-group scale，逻辑 `(k/32, N)`，只供 AIV 反量化使用 |
| yScale | `uint64_t` | `nd_ext_layout_ptn` | `(1, N)` per-channel 输出量化因子，Fixpipe 阶段使用 |
| y（C） | `int8_t` | `nd_ext_layout_ptn` | 量化输出 |

### 缓冲和同步

- 跨核（AIC↔AIV）：CrossCore flag（MODE=4），AIC 侧挂 PIPE_MTE1、AIV 侧挂 PIPE_MTE3；
  `NotifyCube` 的 set flag 排在 MTE3 pipe 的 copy 之后，AIC 看到 ready 时 L1 数据必然就绪。
- AIV 内部：一律用 UB 槽位 mutex（`asc_lock/asc_unlock`）管理 MTE2→V→MTE3 管道
  （input/output 两把锁），无 intra-AIV HardEvent——与跨核 flag 是两套独立硬件资源。
- UB 是 per-core 私有资源（QUAD 两 AIV 各自独立）；跨核共享的是 L1（经
  `GetSharedMemPtr` + CrossCore flag）。
- 输入合法性由 host tiling/API 校验，device 侧不重复校验。

## 模板参数

```cpp
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal;
```

| 参数 | 要求 |
| :--- | :--- |
| `ProblemShape` | `asc::te::shape<int64_t, int64_t, int64_t>`，维序 `(M, N, K)` |
| `BlockMmad` | `MatmulWithWeightQuantPergroup` 特化的 `BlockMmad`；B 侧 tuple 为 `{fp4x2_e2m1_t, ScaleType}`，Bias 槽位传 `void` |
| `BlockEpilogue` | 必须为 `void` |
| `BlockScheduler` | `BlockSchedulerWqmmBlockSplit<ProblemShape, LayoutB, AType>` |

## 特殊数据结构

### `PrologueParams`

```cpp
struct PrologueParams {
    GM_ADDR bGmAddr;
    GM_ADDR scaleBGmAddr;
    uint64_t groupSize;    // 当前仅支持 32
    uint64_t nBubSize;     // AIV 单个 UB 分片的 N 容量
    uint64_t kBubSize;     // AIV 单个 UB 分片的 K 容量
};
```

### `KernelPergroupWeightPrologue::Params`

由 Kernel 从 `mmadParams` + `PrologueParams` 派生（`kBl1Factor`、`kSingleCoreIterNum`、
`kBL1Size` 等 K 分段量），调用方不直接构造。

### `Params`

```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmad::Params mmadParams;
    PrologueParams prologueParams;
    BlockScheduler::Params schedulerParams;
};
```

`mmadParams.l1TileShape` 维序 `(mAL1Size, nBL1Size, kAL1Size, kBL1Size)`，
`l0TileShape` 维序 `(baseM, baseN, baseK)`。

## 特殊成员方法

### GemmUniversal

默认构造；`operator()(const Params&)` 完成 AIC/AIV 分流、Scheduler 构造与 tile 循环。

### KernelPergroupWeightPrologue

```cpp
__aicore__ inline explicit KernelPergroupWeightPrologue(const Params& prologueParams,
                                                        const BlockMmad& sharedMmad);
template <typename GmBTensor_, typename GmScaleTensor_>
__aicore__ inline void operator()(const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale,
                                  const BlockShape& blockShape);
__aicore__ inline void EndSync();
```

`(prologueParams, sharedMmad)` 两参构造引用传入的 BlockMmad：转换后权重 BL1 槽
地址一律经 `sharedMmad.GetSharedMemPtr<WeightOperand>(slotId)` 查询，不在 prologue 侧
复制地址公式。`operator()` 接收 tile 的全 K × validN GM slice（Kernel 已定位到 tile
绝对 N 偏移），跑完该 tile 的 K-factor 写入循环；`EndSync` 把初始多发的 free 标志等
回来，避免残留标志影响下一次运行。

## 执行流程

AIC 侧（`BlockMmad`，细节见配套组件文档）：构造 → `InitAIC` → `InitSync`（预先发出
`BL1Pingpong` 个 free 标志）→ 逐 tile `blockMmad(blockA, blockYScale, blockC, blockShape)`。

AIV 侧：

1. 构造 `BlockMmad` 实例（该构造在 AIC/AIV 上都可执行）与 prologue（UB `UbStorage`
   同时布好
   input/output 槽与 VF scale 掩码 blob）。
2. 逐 tile 调 `blockPrologue(gmBlockWeight, blockScale, blockShape)`，内部按
   `kSingleCoreIterNum` 跑 K-factor 循环，每个 BL1 槽（`kBl1Factor` 个 B 块归并为一槽）
   走"等 free 标志 → 写入 → 发 ready"。
3. 写入分派（`ProduceBL1`）按 `BL1Pingpong` 定型：
   - **QUAD（4）**：`TwoVectorCoreSplit` 把一个 bub 二分给 2 个 AIV——优先沿 N 二分
     （单一维度二分、无循环），N 装不下再沿 K 半区交织（K-split）；
   - **DOUBLE（2）**：两 AIV 按 `GetSubBlockIdx()` 各认领一个 BL1 槽；
   - **单 buffer**：按 `vecCoreParallel` 决定单/双 AIV。
4. 每个分片的三段流水（槽位锁保护）：MTE2 把 packed FP4 与 scale 搬入 UB（ND 经
   `CopyGM2UBWeight` tile，NZ 直接走标准 `copy_gm_to_ub`）→ V 执行
   `Dequant<A8W4TcgDequantParams<32>>::Run`（FP4×scale→FP8）→ MTE3 经
   `CopyPaddedUBToL1` 转换权重族把转换结果搬入共享 BL1 槽。
5. 全部 tile 完成后 `EndSync` 等回剩余的 free 标志。

## 格式和尾块

- **ND（`dn_ext_layout_ptn`，B 转置）**：UB 内 FP4 按 64 对齐行 pitch，scale 按 32B 对齐；
  转换输出 `Weight8BitDnToZnUbLayoutPtn`（K32 slab + N 列 padding），UB→L1 剥 gap 进
  标准 ZN。
- **NZ（`nz_layout_ptn`）**：N 偏移全程 32 对齐（fractal 寻址不变式）；UB 内为紧凑
  pitch 的 NZ pattern；转换输出 `NzRowPaddingLayoutPtn`（256 元素 chunk + ping-pong
  交织），UB→L1 压回 NZ fractal。
- **尾 K 组**：`groupNum` 与相关拷贝 chunk 数均取 floor（`validK / 32`），尾组不转换。
- **K-split**：QUAD 且 K 装不下时的两 AIV K 半区交织，其 UB→L1 搬运保留 ops-nn 原始
  手拼 pattern（标准 `copy_ub_to_l1` 直调），协议算术受保护不重写。

## 调用示例

以下类型组合与 nn 仓 `quant_batch_matmul_v4` 的 blaze 入口一致（`m`、`n`、`k` 为
`int64_t`，tiling 字段来自 host tiling）：

```cpp
using AType = fp8_e4m3fn_t;
using BType = fp4x2_e2m1_t;
using ScaleType = bfloat16_t;                 // 或 half
using CType = int8_t;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::zn_layout_ptn;       // ND 权重时改为 dn_ext_layout_ptn
using LayoutC = asc::te::nd_ext_layout_ptn;
using BTypeTuple = AscendC::Std::tuple<BType, ScaleType>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::MatmulWithWeightQuantPergroup, AType, LayoutA, BTypeTuple, LayoutB,
    CType, LayoutC, void, void>;              // T-CG 无 bias，槽位传 void
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerWqmmBlockSplit<
    ProblemShape, LayoutB, AType>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;

typename BlockMmad::Params mmadParams{};
mmadParams.aGmAddr = x1;
mmadParams.cGmAddr = y;
mmadParams.yScaleGmAddr = yScale;
mmadParams.l1TileShape = asc::te::make_shape(mAL1Size, nBL1Size, kAL1Size, kBL1Size);
mmadParams.l0TileShape = asc::te::make_shape(baseM, baseN, baseK);
mmadParams.vecCoreParallel = vecCoreParallel;
mmadParams.AL1Pingpong = al1Pingpong;
mmadParams.BL1Pingpong = bl1Pingpong;
mmadParams.dbL0C = dbL0C;

Kernel::Params params{
    asc::te::make_shape(m, n, k),
    mmadParams,
    {x2, x2Scale, groupSize, nBubSize, kBubSize},
    {cubeNumBlocksM, cubeNumBlocksN, baseM, baseN, iterateOrder}};
Kernel kernel;
kernel(params);
```

## 适用场景

- arch35 T-CG per-group A8W4（A=FP8、B=packed FP4、per-group FP16/BF16 scale）量化
  matmul，输出 int8，per-channel yScale 量化。
- Weight NZ 或 Weight ND 输入；`QuantBatchMatmulV4` 集成中的单 Batch `(M, N, K)` 推理路径。
