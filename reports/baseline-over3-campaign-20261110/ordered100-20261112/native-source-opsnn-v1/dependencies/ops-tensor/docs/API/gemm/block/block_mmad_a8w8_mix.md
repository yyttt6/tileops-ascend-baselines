# Block Mmad A8W8 Mix
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_a8w8_mix.h)

## 功能说明
MIX 模板 A8W8 量化矩阵乘 Block，基于 Tensor API，仅在 **AIC** 上执行。完成 int8 × int8 的 int32 累加，并通过 fixpipe（NoQuant，原始拷贝）把 L0C 结果搬到 UB；**本层不做 scale/bias**，反量化由 AIV 侧的 [block_epilogue_dequant](../../epilogue/block/block_epilogue_dequant.md) 完成。支持 ND / WeightNz（FRACTAL_NZ）权重布局、L1/L0 双缓冲、A 矩阵全载模式。

**继承自**：[Block Mmad 基础框架](./block_mmad.md)

## 特殊约束

### 调度策略限制
仅特化于 `MatmulWithScaleMix<A_FULL_LOAD_MODE, ATOMIC_ADD>`（ScheduleType = `KernelMmadWithScaleMix`）。不支持 `MatmulMultiBlockBasic` / `MatmulMultiBlockWithStreamK` / `MatmulWithScaleMx`。

### 量化数据类型
- A/B：int8（A8W8）。
- L0C 累加类型 `L0CType = GetMmDstType<AType>::Type`（int8 → int32）。
- C0_SIZE = `asc::te::c0_element<int8_t>`（32）；C0_SIZE_L0C = 16。

### 计算模式与输出目标
- 仅 AIC 模式。
- 输出走 **L0C → UB**（`CopyL0C2UB`，`FINAL_ACCUMULATION` + `DUAL_DST_SPLIT_M` trait），**不写 GM、不支持 workspace**。最终 GM 写由 AIV epilogue 负责。

### Scale / Bias
本层不处理 scale 与 bias。`Params` 保存 A/B 的 GM 地址及 L1/L0 切分参数；输出 Tensor 由 `operator()` 的 `ubC` 参数传入。

## 特殊静态常量
| 常量 | 说明 |
|------|------|
| WEIGHT_NZ | B 是否为 NZ 格式（`IsWeightNz<LayoutB>`） |
| TRANS_A / TRANS_B | A/B 是否转置 |
| C0_SIZE | C0 对齐大小（int8: 32） |
| C0_SIZE_L0C | L0C 的 C0（16），定义于 `common_utils.h` |
| DOUBLE_BUFFER_COUNT | 双缓冲数量（2），定义于 `common_utils.h` |

## 特殊类型别名
| 类型 | 说明 |
|------|------|
| BType | 权重类型，取自 `BTypeTuple` 的第 0 元素 |
| X2ScaleType | x2Scale 类型，取自 `BTypeTuple` 的第 1 元素 |
| L0CType | L0C 累加类型（int8 → int32） |
| MakeLayoutAL1 / MakeLayoutBL1 | 据 TRANS_A/TRANS_B 选择 ZN/NZ 的 L1 FrameLayout |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR aGmAddr{nullptr};   // A 矩阵 GM 起始地址
    GM_ADDR bGmAddr{nullptr};   // B 矩阵 GM 起始地址
    ProblemShape problemShape;  // 问题规模 (m, n, k, batch)
    BlockShape l0TileShape;     // L0 tile (baseM, baseN, baseK)
    uint64_t kAL1{0};           // A 的 L1 K 轴切分
    uint64_t kBL1{0};           // B 的 L1 K 轴切分
    uint64_t l1BufferNum{0};    // L1 缓冲数量
    bool enableL0CPingPong{false}; // 是否启用 L0C 双缓冲
};
```

## 特殊成员方法

### 构造/析构函数
```
__aicore__ inline BlockMmad()    // SetFlag MTE1_MTE2 ×4，开启 MMLayoutTransform
__aicore__ inline ~BlockMmad()   // WaitFlag MTE1_MTE2 ×4，关闭 MMLayoutTransform
```

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
功能：读取问题规模与 tile，按全载模式和缓冲数量计算单个 A/B L1 Buffer 的局部大小，并调用 `GetL1BufferOffset(aL1OneBuffer, bL1OneBuffer)` 计算 L1 Buffer 偏移。
说明：
- `FULL_LOAD_MODE == A_FULL_LOAD_MODE`：A 矩阵全载，`aL1OneBuffer = CeilAlign(baseM, ...) * CeilAlign(k_, ...)`。
- `aL1OneBuffer` / `bL1OneBuffer` 是 `Init()` 内的局部变量，通过参数传给 `GetL1BufferOffset`，不作为类成员保存。

### operator函数
```
template <typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void operator()(
    TensorA gmA,            // A 矩阵 GM Tensor（当前 tile）
    TensorB gmB,            // B 矩阵 GM Tensor（当前 tile）
    TensorC ubC,            // L0C 结果输出到 UB 的 Tensor
    BlockShape singleShape) // Tile 形状
```
执行流程：
1. 据 `kAL1_` / `kBL1_` 关系选择迭代策略：`IterateABL1` / `IterateAL1BL1` / `IterateBL1AL1`。
2. 在 L0C（NZLayout，`CeilAlign(curM, 2)` 对齐）完成 int32 累加。
3. `CopyL0C2UB`（NoQuant，FINAL_ACCUMULATION，DUAL_DST_SPLIT_M）把 L0C 搬到 `ubC`。
4. `enableL0cPingPong_` 时切换 L0C ping-pong。

## 数据流
```
GM(int8 A) ─┐
GM(int8 B) ─┴→ L1(NZ/ZN) → L0A/L0B → L0C(int32 累加) ─fixpipe NoQuant→ UB(int32)
```
> 后续 dequant（× x2Scale [× x1Scale] + bias → bf16/fp16/fp32）由 AIV epilogue 完成。

## 性能优化建议
- `l1BufferNum`：2 或 4，平衡 L1 占用与流水线并行度。
- A 全载模式：A 常驻 L1，适用于大 K、小 M。
- `dbL0C`：启用 L0C 双缓冲，重叠 AIC 累加与 AIV 后处理。

## 适用场景
- int8 量化矩阵乘的 AIC 计算层。
- ND / WeightNz（FRACTAL_NZ）布局；配合 MIX Kernel + dequant epilogue 使用。
