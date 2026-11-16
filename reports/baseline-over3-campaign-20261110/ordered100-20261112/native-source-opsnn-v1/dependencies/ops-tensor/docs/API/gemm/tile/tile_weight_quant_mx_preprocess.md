# Tile Weight Quant MX Preprocess
> 相关头文件：
> [数据搬运聚合头 datamove.h](../../../../include/blaze/gemm/tile/datamove.h)（含 GM→UB、UB→L1） ·
> [计算原语聚合头 compute.h](../../../../include/blaze/gemm/tile/compute.h)（含 FP4→FP8、Bias 缩放）

## 功能说明

这些 Tile 仅描述 arch35 的 packed FP4 权重前处理，不负责 Kernel 的 AIV/AIC 跨核同步或 L1
buffer 生命周期。Kernel 负责同步和存储管理，Tile 负责 Tensor layout、FP4→FP8 转换和数据搬运。

## 特殊约束

### 架构和数据类型

- 仅支持 `__NPU_ARCH__ == 3510`（arch35）。
- 输入权重为 `fp4x2_e2m1_t`，转换输出为 `fp8_e4m3fn_t`。
- bias 支持 Kernel 配置的输出类型（通常为 `half` 或 `bfloat16_t`）。

### Layout

| Tile | 输入 layout | 输出 layout | 说明 |
| :--- | :--- | :--- | :--- |
| `CopyGM2UBWeight` | `zn_layout_ptn` / `dn_ext_layout_ptn` | 带显式 stride 的 `zn_layout_ptn` / packed `dn_ext_layout_ptn` | 根据 Weight NZ/ND layout 选择搬运路径 |
| `ShiftW4ToW8` | `zn_layout_ptn` / `dn_ext_layout_ptn` | `Weight8BitZnToZnUbLayoutPtn` / `Weight8BitDnToZnUbLayoutPtn` | 根据输入 format 选择 VF 路径 |
| `CopyUB2L1Weight8Bit` | `Weight8BitZnToZnUbLayoutPtn` / `Weight8BitDnToZnUbLayoutPtn` | L1 ZN | ZN 直接压实，DN→ZN 剥离 gap |

## 使用方式

### `CopyGM2UBWeight`

```cpp
auto copy = asc::te::make_copy(Blaze::Gemm::Tile::CopyGM2UBWeight{});
asc::te::copy(copy, weightUbTensor, weightGmSlice);
```

仅接受 `zn_layout_ptn` 或 `dn_ext_layout_ptn` 的源 Tensor。packed FP4 的 K 方向按两个 4-bit
元素一个字节处理，GM/UB 步长（stride）由 Tensor layout 提供。

### `ShiftW4ToW8`

```cpp
Blaze::Gemm::Tile::ShiftW4ToW8<fp8_e4m3fn_t, fp4x2_e2m1_t>(weight4Ub, weight8Ub);
```

以类构造函数形式发起转换。源为 Weight NZ 对应的 `zn_layout_ptn` 时执行 ZN interleave 转换并生成
`Weight8BitZnToZnUbLayoutPtn`；源为 `dn_ext_layout_ptn` 时按 N 行读取并执行 DN→ZN-like 的
`DATA_BLOCK_COPY` 输出，生成 `Weight8BitDnToZnUbLayoutPtn`。两条路径都由输入 format 对应的
layout 在编译期选择。

MX GMM 融合路径将 bias 预缩放合并到同一条 VF 命令，`processBias` 在编译期分派，
steady-state K 迭代中不含 bias 分支或额外 VF 入口：

```cpp
Blaze::Gemm::Tile::ShiftW4ToW8<fp8_e4m3fn_t, fp4x2_e2m1_t>(
    weight4Ub, weight8Ub, biasInUb, biasOutUb, /*processBias=*/true);
```

### `CopyUB2L1Weight8Bit`

```cpp
auto copy = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
asc::te::copy(copy, weightL1Tensor, weight8UbTensor);
```

`Weight8BitUBLayout` 的原有两参数调用现在返回 `Weight8BitZnToZnUbLayoutPtn`，并委托到
`Weight8BitZnToZnUBLayout`；后者供 kernel 传入运行时的 interleave stride。新增的
`Weight8BitDnToZnUBLayout` 返回
`Weight8BitDnToZnUbLayoutPtn`，对应 DN→ZN 的额外 gap 搬运。原始 GM 权重是 ND 还是 NZ
由 W4→W8 阶段选择对应的物理 UB layout。

### `ScaleMxBias`

```cpp
Blaze::Gemm::Tile::ScaleMxBias<BiasType>(biasInUbTensor, biasOutUbTensor);
```

以类构造函数形式发起计算。该 Tile 将 bias 乘以 MX MMAD 所需的 `1/64`。它以 256B 为一轮并使用整向量 mask，**输入和输出
UB backing storage 均至少需要 `CeilAlign(N, 256 / sizeof(BiasType))` 个元素**；Tensor 的有效列数可为
`Align16(N)`。

## 布局契约

- 算子 ND 格式的 B 是转置权重；Blaze 使用 `dn_ext_layout_ptn`，Tensor 坐标顺序为 `(K, N)`，
  GM 物理数据按 `(N, K)` 行主序存放。
- DN→ZN 转换后的临时 UB layout 使用 `Weight8BitDnToZnUbLayoutPtn`，逻辑 shape 仍为 K×N，
  每个 K32 slab 的物理 N span 为 `(Align16(N) + 1) * 32B`。额外的 `+1` 表示一个 32B data
  block，用于打散相邻 K32 slab 的 UB bank 映射，并不是对 GM 逻辑 N 轴增加有效元素。
- UB→L1 时，`CopyUB2L1Weight8Bit` 根据 layout stride 计算并剥离该 UB-only gap，L1 保持标准 ZN。
- Weight NZ 转换后的 UB 使用标准 `zn_layout_ptn` 配合显式 shape/stride，取 `n0=8`、
  `n1=Align16(N)/8`，因此 footprint 按 `Align16(N)` 计算。

## 数据流

```text
GM packed FP4
    │ CopyGM2UBWeight
    ▼
UB packed FP4 ── ShiftW4ToW8 ──► UB FP8（ZN 或 DN→ZN pitched layout）
                                      │ CopyUB2L1Weight8Bit
                                      ▼
                                  L1 ZN weight

UB bias ── ScaleMxBias ──► UB bias(1/64) ── Tensor API copy_ub_to_l1 ──► L1 bias
```

调用方应包含聚合头 `blaze/gemm/tile/datamove.h`（数据搬运原语）和
`blaze/gemm/tile/compute.h`（计算原语），不应直接包含 `tile/arch35` 实现文件。

## 使用场景

由 [Kernel Matmul Mix Weight Prologue](../kernel/kernel_matmul_mix_weight_prologue.md) 的 AIV
前处理路径调用；不应脱离该 Kernel 的 buffer 所有权和 ready/free 标志协议单独组合。
