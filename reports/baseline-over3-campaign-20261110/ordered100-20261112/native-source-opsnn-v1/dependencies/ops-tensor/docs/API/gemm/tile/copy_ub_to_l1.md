# Copy UB To L1

> [代码位置](../../../../include/blaze/gemm/tile/arch35/copy_ub_to_l1.h)

## 功能说明

`CopyPaddedUBToL1` 按**源 UB layout pattern** 在编译期分派，覆盖两族场景：
FP16/BF16 权重 padding 布局（WQMM 权重反量化）与转换后 8-bit 权重（T-CG per-group A8W4）。
分支选择完全由 Tensor 类型在编译期完成，`Copy` 内不做运行时判断。

## FP16/BF16 权重 padding 布局

### 功能说明

该组件将带分组填充的 FP16/BF16 权重从 UB 搬运到 L1，供 WQMM 权重反量化使用。
调用方通过 `asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{})` 创建 Copy atom，
再使用 `asc::te::copy(copyUB2L1, dstTensor, srcTensor)` 执行搬运。
`CopyPaddedUBToL1` 根据源布局 pattern 选择对应的搬运分支。
该实现直接从源、目标的层次化布局提取搬运参数，调用 `asc_copy_ub2l1` 完成数据搬运。

### 使用限制

- 仅在 `__NPU_ARCH__ == 3510` 时由聚合头 `blaze/gemm/tile/datamove.h` 引入。
- 源 Tensor 位于 UB，目标 Tensor 位于 L1；两端元素类型相同，支持 FP16 或 BF16。
- 源布局 pattern 为 `Blaze::Gemm::zn_row_padding_layout_ptn` 或
  `Blaze::Gemm::nz_col_padding_layout_ptn`。
- 源 pattern 决定搬运分支；目标布局提供对应分组轴的实际 L1 行距。两端完整 shape 可以不同，
  但每个分组内的数据排列必须兼容。
- 调用方保证范围非空、地址及连续段长度和行距按 32 字节对齐、段间空隙非负，
  搬运参数可由指令字段表示，且缓冲区容量覆盖包含填充的物理范围。

### 布局构造接口

源布局由 [layout_struct.h](../../../../include/blaze/gemm/utils/layout_struct.h) 中的两个辅助类构造：

```cpp
template <typename T>
struct ZnRowPaddingUBLayout {
    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize, int64_t groupPitch) const;
};

template <typename T>
struct NzColPaddingUBLayout {
    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize, int64_t groupPitch) const;
};
```

两个类均位于 `Blaze::Gemm` 命名空间，`T` 为 FP16 或 BF16。
`kSize`、`nSize` 是权重逻辑坐标 `(K,N)` 下的有效长度；`groupPitch` 为包含填充的实际 UB 分组行距，
三者均以元素计。令 `C0=32/sizeof(T)=16`，构造结果如下：


| 构造类                    | 源 pattern                  | shape                             | stride，单位为元素         |
| ------------------------- | --------------------------- | --------------------------------- | -------------------------- |
| `ZnRowPaddingUBLayout<T>` | `zn_row_padding_layout_ptn` | `((16,ceil(kSize/16)),(1,nSize))` | `((1,groupPitch),(16,16))` |
| `NzColPaddingUBLayout<T>` | `nz_col_padding_layout_ptn` | `((1,kSize),(16,ceil(nSize/16)))` | `((16,16),(1,groupPitch))` |

ZN 保留有效 N 长度并对齐 K，NZ 保留有效 K 长度并对齐 N。
WQMM VF 的 UB 分组行距为 `65*16` 个元素，尾块也使用该行距，不能根据有效 shape 缩小间距。

### 搬运参数

对于源 shape `((k0,k1),(n0,n1))`，按源 pattern 提取以下参数。
行距分别从源、目标 layout 的对应 stride 中读取，单位为元素。


| 源 pattern                  | 连续段数量 | 每段元素数 | 分组行距       |
| --------------------------- | ---------- | ---------- | -------------- |
| `zn_row_padding_layout_ptn` | `k1`       | `k0*n0*n1` | `stride[0][1]` |
| `nz_col_padding_layout_ptn` | `n1`       | `k0*k1*n0` | `stride[1][1]` |

设每段元素数为 `segmentElements`，两端行距为 `srcPitch`、`dstPitch`。
指令的 `block_len`、`src_gap` 和 `dst_gap` 以 32 字节为单位：

```text
block_len = segmentElements * sizeof(T) / 32
src_gap   = (srcPitch - segmentElements) * sizeof(T) / 32
dst_gap   = (dstPitch - segmentElements) * sizeof(T) / 32
```

### 使用方式

调用方包含 `blaze/gemm/tile/datamove.h`，使用上述辅助类构造源布局，并为目标提供兼容的 L1 布局。
以下展示 ZN 分组的调用片段，`ubWeight` 和 `l1Weight` 为已分配的类型化指针，
`l1Layout` 描述目标 L1 的实际分组排列和行距。

```cpp
auto ubLayout = Blaze::Gemm::ZnRowPaddingUBLayout<T>{}(kSize, nSize, groupPitch);
auto srcTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub>(ubWeight), ubLayout);
auto dstTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::l1>(l1Weight), l1Layout);
auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{});
asc::te::copy(copyUB2L1, dstTensor, srcTensor);
```

NZ 分组使用 `NzColPaddingUBLayout<T>` 构造源布局，并提供对应的 L1 NZ 布局。
`CopyPaddedUBToL1` 通过 `copy_traits` 注册，默认使用 Tensor API 的 `ub_to_l1_trait_default`。

## 转换后 8-bit 权重

### 功能说明

转换后 8-bit 权重的 UB → L1 搬运分支（`CopyPaddedUBToL1` 转换权重族）。向量单元反量化输出的
UB 排布自带 padding 或 chunk 交织（对齐、bank 打散、ping-pong 交织），L1 侧却是标准
fractal；本 op 按**源 UB layout** 分三个实现分支，把带 padding 的源压回 L1 目标 layout。

分支选择完全由 Tensor 类型在编译期完成，`Copy` 内不做运行时判断。

### 特殊约束

#### 架构和数据类型

- 实现位于 `tile/arch35/copy_ub_to_l1.h`，仅 `__NPU_ARCH__ == 3510`；调用方应包含聚合头
  `tile/datamove.h`。
- 源/目标元素均须为 1 字节（转换后的 8-bit 权重），static_assert 门禁。

#### 分支与 layout


| 源 UB layout                                            | 实现               | L1 目标 layout  | 语义                                                                                                                                |
| :------------------------------------------------------ | :----------------- | :-------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| `Weight8BitDnToZnUbLayoutPtn` / `ZnColPaddingLayoutPtn` | `CopyZnColPadding` | `zn_layout_ptn` | ZN 列 padding：k1 个 K-slab，每 slab 含 nSize 个相邻 32B 块（每 N 列 32 个连续 K），slab pitch 携带 gap；搬运时两侧按 stride 剥 gap |
| `Weight8BitZnToZnUbLayoutPtn`                           | `CopyZnToZnWeight` | `zn_layout_ptn` | ZN→ZN：k1×n1 个 fractal 块，源块间 stride 剥离，L1 侧连续                                                                         |
| `NzRowPaddingLayoutPtn`                                 | `CopyNzRowPadding` | `nz_layout_ptn` | NZ 行 padding：N1 fractal × 每 group 32K×32N slab × 256 元素 chunk（跨 BL1 ping-pong 交织），目标继承 AIC 读侧 frame 的 n1 pitch |

`Weight8BitDnToZnUbLayoutPtn` 是 旧版的legacy tag命名方式，由于功能已经上库，为了避免对外的功能问题，先不做删除`ZnColPaddingLayoutPtn` 是相同模式的新式命名tag，两者路由到同一分支，后续工作中会逐渐把旧的命名往新命名切换。

#### NZ 族 layout

`NzRowPaddingLayout`（`blaze/gemm/utils/layout_struct.h`）描述 VF 输出的 NZ 族物理排布：
最外到最内为 N1 fractal（32 N）、K group（32 K）、group 内 vector loop、256 元素
vector chunk；`InnerStride` 参数携带相邻 chunk 间的 ping-pong 交织步长。该 layout 是
VF 输出排布的单一事实源：[Dequant](../../epilogue/tile/arch35/dequant.md) 的 NZ 输出与
本 op 的 `CopyNzRowPadding` 源两侧共用。`kGroupNum = kSize / 32`（floor）带 per-group
印记，非 group-32 场景复用时需参数化 GROUP_SIZE。

### 特殊类型

#### `CopyPaddedUBToL1` 转换权重族签名

```cpp
struct CopyPaddedUBToL1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src);
};
```

`dst` 为 L1 tensor（ZN 族 `zn_layout_ptn` / NZ 族 `nz_layout_ptn`），`src` 为上表所列
UB layout 之一的 tensor；shape/stride 全部从两侧 layout 读取，无额外参数。

`CopyNzRowPadding` 内含 fast path：当 `dstN1Stride == blocksPerN1 * chunkLen`（覆盖的
chunk 恰好连续铺满目标 fractal，即无尾 K 组残缺）时，整块合成一次 gap-0 的搬运，
否则退化为逐 N1 fractal 一，两侧各按 layout stride 推进。

#### 兼容转发层

`tile/arch35/copy_weight_ub_to_l1.h` 的旧 op `CopyUB2L1Weight8Bit` 是纯转发层，仅转发到
`CopyPaddedUBToL1::Copy` 的转换权重分支，存量 `make_copy(CopyUB2L1Weight8Bit{})` 调用方无需改动。

### 使用方式

```cpp
// 新式入口
auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{});
asc::te::copy(copyUB2L1, weightL1Tensor, weightOutUbTensor);

// legacy 入口（转发，等价）
auto copyLegacy = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
asc::te::copy(copyLegacy, weightL1Tensor, weightOutUbTensor);
```

底层原语为 `asc_copy_ub2l1`，blockLen/gap 参数均以 32B 块为单位从 layout stride 折算。

### 适用场景

- [Kernel Wqmm Mix Pergroup](../kernel/kernel_wqmm_mix_pergroup.md) AIV prologue 的
  `CopyVecOut2L1Nd/Nz` 路径：把 `Dequant` 输出的 UB 权重搬入跨核共享 BL1 槽。
- K-split 场景不走本 op：两 AIV K 半区交织的 UB→L1 协议保留 ops-nn 原始手拼 pattern
  （标准 `copy_ub_to_l1` trait 直调，见 kernel 的 `CopyVecOut2L1NzKSplit`）。
