# Block Mmad Wqmm Mix Prologue Fixpipe Quant
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_wqmm_mix_prologue_fixpipe_quant.h)

## 功能说明

这是 `BlockMmad<MatmulWithWeightQuantPergroup, ...>` 的 arch35 特化，负责 AIC（Cube）侧
的单 tile T-CG per-group 量化 matmul：等待 AIV（Vector）把 FP4 权重反量化成 FP8 写入共享
L1 后，搬入 L0 执行 FP8×FP8 MMAD，最后 Fixpipe 阶段乘 per-channel yScale 量化输出
（`out = (x1 @ (x2 ⊙ x2Scale)) ⊙ yScale`）。

本组件不接收 B Tensor——B 的 L1 数据由 Kernel 内的 AIV prologue 写入；A/YScale/C 是
Kernel 准备的 per-tile GM slice。转换后权重的 BL1 槽位也由本组件统一划分：AIV 侧通过
`GetSharedMemPtr` 查询这些共享地址。

**配套组件**：[Kernel Wqmm Mix Pergroup](../kernel/kernel_wqmm_mix_pergroup.md) 和
[Dequant](../../epilogue/tile/arch35/dequant.md)

## 特殊约束

### 架构和 policy

仅支持 `__NPU_ARCH__ == 3510`（arch35），且只能与 `MatmulWithWeightQuantPergroup` 组合
（`blaze/gemm/policy/dispatch_policy.h`）。该 policy 内嵌 `SyncProtocol`：跨核
ready/free 标志使用 CrossCore flag（MODE=4，`AIV_READY_FLAG=1`、`AIC_FREE_FLAG=2`、
影子 id 偏移 `FLAG_ID_MAX=16`），block 侧经 `using SyncProtocol = typename
DispatchPolicy::SyncProtocol;` 透传。

### 模板类型约束

- `BTypeTuple` 为二元 tuple `{BType, ScaleType}`：B 必须为 packed FP4，ScaleB
  （`half`/`bfloat16_t`）只在 AIV 反量化时使用，不进入 AIC。
- A 必须为 `fp8_e4m3fn_t` 且非转置 ND layout。
- 尾部 `BiasType/LayoutBias` 槽位仅是签名占位，必须传 `void`：T-CG 无 bias 输入。

### 缓冲和同步

- L1 总预算 512KB，`L1Storage` 按传入的 `BL1Pingpong/AL1Pingpong/vecPingpong` 组合规划
  B/A 槽偏移（QUAD 模式下 B 槽 0/2 与 1/3 分居两个 256KB 半区，A 槽交错其间）。
- B 的 L1 数据由 AIV 写入、AIC 读取：`InitSync` 启动时先发出 `BL1Pingpong` 个 AIC_FREE
  标志，相当于预先声明这些槽位是空的；之后 AIV 每写完一个槽要先等到一个 AIC_FREE，
  AIC 每读完一个槽再发回一个 AIC_FREE；`EndSync` 结束前把最初多发出的那几个标志等
  回来。这样收尾时置位与等待次数刚好配平：AIC 一定读完了 AIV 写入的最后一批数据，
  也不会留下残留的标志位干扰下一次运行。
- 跨核等待/通知按 `BL1Pingpong` 模式选择标志组：QUAD 等/发 BOTH（影子 id + 基础
  id，各配对一个 AIV）；DOUBLE 按 slot 奇偶选 BASE_ONLY/OFFSET_ONLY；单 buffer 按
  `vecCoreParallel` 区分单 AIV（BASE_ONLY）与双 AIV（BOTH）。
- A 的 GM→L1/ L1→L0 / L0C→GM 搬运由本组件负责；intra-AIC 流水经
  `BufferManager` 的槽位锁保护。

## 特殊数据结构

### `Params`

```cpp
struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR cGmAddr;
    GM_ADDR yScaleGmAddr;
    L1TileShape l1TileShape;   // (mAL1Size, nBL1Size, kAL1Size, kBL1Size)
    L0TileShape l0TileShape;   // (baseM, baseN, baseK)
    uint8_t vecCoreParallel;
    uint16_t AL1Pingpong;
    uint16_t BL1Pingpong;      // 限 4 或 2（NZ）/ 4、2、1（ND）
    uint32_t dbL0C;
};
```

| 字段 | 说明 |
| :--- | :--- |
| `aGmAddr` / `cGmAddr` / `yScaleGmAddr` | A、输出 C、per-channel yScale 的 GM 地址；yScale 为 `(1, N)` 的 uint64 行 |
| `l1TileShape` | L1 级 A/B 块尺寸；`kAL1Size` 是 L1 预算内 A 的自由大块，`kBL1Size` 受 AIV 产能约束偏小，典型 `kAL1Size ≥ kBL1Size` |
| `l0TileShape` | L0 级基准 tile |
| `vecCoreParallel` / `AL1Pingpong` / `BL1Pingpong` / `dbL0C` | 来自 host tiling 的流水配置 |

### 协作接口

- `GetSharedMemPtr<WeightOperand>(slotId)`：纯地址查询，返回 BL1 槽相对 L1 起始的 byte
  偏移；`slotId` 与 AIV 侧使用相同映射且小于 `BL1Pingpong`。跨核 ready/free 标志与此
  无关，由 SyncProtocol 单独管理。
- `WeightOperand`：空的结构体标签（仅用于类型区分），当前共享数据只有 weight 一种。
- `SyncProtocol`：见"架构和 policy"。

它们是 Kernel prologue 与 BlockMmad 之间的内部协作契约，调用方通过 Kernel 使用，不应
自行创建第二套 L1 地址或同步协议。

## 特殊成员方法

### 构造函数和析构函数

```cpp
__aicore__ inline explicit BlockMmad(const Params& params);
__aicore__ inline ~BlockMmad();
```

构造只做布局计算和 `L1Storage` 初始化，不含任何只属于 AIC 的操作（`SetMMLayoutTransform`
内部已按核类型区分，仅在 AIC 上生效），所以 AIC 和 AIV 都能执行同一个构造——配对的
AIV 侧构造相同实例，即可查询到与 AIC 一致的共享地址。析构时关闭 MM layout transform。

### `InitAIC` / `InitSync`

```cpp
__aicore__ inline void InitAIC();    // AIC 专属：初始化 L1A/BL1/L0/L0C/ScaleL1 槽位
__aicore__ inline void InitSync();   // AIC 专属：预先发出 BL1Pingpong 个 free 标志
```

### `operator()`

```cpp
template <typename TensorA_, typename TensorYScale_, typename TensorC_>
__aicore__ inline void operator()(const TensorA_& gmBlockA, const TensorYScale_& gmBlockYScale,
                                  const TensorC_& gmBlockC, const BlockShape& blockShape);
```

三个 Tensor 为当前 scheduler tile 的 GM slice（A: `(validM, k)`，YScale: `(1, validN)`，
C: `(validM, validN)`）；一次调用处理一个 M/N tile，K/AL1/BL1/L0 循环在内部完成。

## 计算流程

1. 记录 tile 的 `validM/validN`，按 `kSingleCoreIterNum`（K 按 `min(kAL1Size, kBL1Size)`
   分段数）进入 K 生成循环。
2. 外层生成循环（步进 `kAl1Factor`）表达"A 槽驻留期"：每代先把 A 的当前 K 段
   ND2NZ 搬入 L1（`GetAL1`），随后在内层按 `kBL1Size` 逐 B 块推进。
3. 内层每个 `kFactorIdx`：`WaitBL1`（B 块整倍数边界处等 AIV 的 ready 标志）→
    `IterateMatmul`（L1→L0 搬运 + MMAD 累加到 L0C；从共享 BL1 槽的 intra-slot 偏移读
    转换后 B）→ `PostProcess`（B 块整倍数边界处发回 free 标志、推进 BL1 槽）。
4. K 循环结束后 `FixpipeOutput`：yScale 搬入 L1 尾部，L0C 经 `copy_l0c_to_gm` 携带
   scale 量化写回 C 的 GM slice；末代/末块双尾截断对齐在循环边界处理。

A 的生成循环只为把"一个 A 槽的整个使用周期"圈进 bufMgr 槽位锁的作用域，B 的驻留
用一个取模判断控制重载，两级循环与 ops-nn 旧实现的逐拍推进等价。

## Kernel 内部调用

该 Block 不能脱离配套的 AIV 写入方独立调用：执行时按相同 tile 序列等 AIV 写入 B 并发
ready 标志，AIV 也依赖其 `GetSharedMemPtr` 查询 BL1 地址。完整类型组装和参数构造见
[Kernel Wqmm Mix Pergroup](../kernel/kernel_wqmm_mix_pergroup.md#调用示例)。

```cpp
BlockMmad blockMmad(params.mmadParams);
blockMmad.InitAIC();
blockMmad.InitSync();
for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
    // scheduler tile → blockA/blockYScale/blockC GM slice
    blockMmad(blockA, blockYScale, blockC, blockShape);
}
```

## 适用场景

- `GemmUniversal` Weight-Quant-Pergroup 特化的 AIC 计算侧。
- 配套 Kernel 支持 arch35 T-CG per-group A8W4 Weight ND/NZ 输入，AIV 负责反量化
  前处理，输出经 per-channel yScale 量化。
- 不适合单独调用；没有配套的 AIV 写入方时，B 的 L1 数据和 ready 标志都不存在。
