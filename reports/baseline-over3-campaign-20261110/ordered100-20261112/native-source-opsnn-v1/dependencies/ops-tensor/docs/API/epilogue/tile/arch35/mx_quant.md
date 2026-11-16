# MxQuant（Tile 级 MX 动态量化链）
> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/mx_quant.h)

## 功能说明

Tile 级 MX（microscaling）动态量化链组件，将 Block Epilogue 中重复的量化链
（eMax 提取 → E8M0 scale/倒数计算 → FP8/FP4 量化 → scale 布局转置）收敛为单一
实现，供多个 Block Epilogue 复用。每个阶段是独立的公共入口：

- **GroupMaxExp**：逐 32 元素组提取 bf16 指数域最大值（OCP）或绝对值最大码点
  （cuBLAS / DynDtypeRange），每组产出一条 uint16 maxExp
- **GenScale**：按算法（OCP / cuBLAS / DynDtypeRange）计算 E8M0 yScale
  （`DIST_PACK_B16` 打包写入）与 uint16 reciprocal（供 Quantize 乘法）
- **Quantize**：src × 逐组 reciprocal → 饱和 FP8/FP4 cast，`DIST_PACK4_B32`
  打包写入 int8 buffer；FP4 舍入模式运行时指定
- **TransScaleLayout**：打包 yScale → 每行 32B 行块的 MTE 搬运布局
- **TransFp4OutLayout**（可选，仅 FP4 输出）：FP4 打包输出从 Align16(n/2) 字节
  行距重排为 Align32(n/2) 字节行距

采用两层设计：
- **公共接口（`__aicore__`）**：接收 `make_tensor` 构造的 UB 张量，内部经
  `.data().get()` 提取 `__ubuf__` 指针，loop 计数由 totalCount / totalScale
  内部推导（调用方不再传 loopCount），打包 VfParams 后经 `asc_vf_call` 委托；
  入口含 static_assert 类型/排布门禁与 `if ASCEND_IS_AIC` 早退
- **私有实现（`__simd_vf__` / `__simd_callee__`）**：Reg API 寄存器级计算。
  OCP 与 DynDtypeRange 的 reciprocal 尾段（Select nan/zero/special 三连）
  指令完全一致，共享 `ScaleReciprocalTailCallee`

## 特殊约束

### 架构
仅 dav-3510（`__NPU_ARCH__ == 3510`），经
[compute.h](../compute.md) 派发引入，Block 层禁止直接 include 本头文件。

### 数据类型
`DataTypeOut` 支持 `fp8_e4m3fn_t` / `fp8_e5m2_t` / `fp4x2_e2m1_t` /
`fp4x2_e1m2_t`（类级 static_assert 门禁）。FP8 与 FP4 量化路径由 `DataTypeOut`
编译期分派（`if constexpr`），FP4 舍入模式（RINT / FLOOR / ROUND）由
`Quantize` 的运行时参数分派到模板 Vf。

### 运行时算法配置（MxQuantConfig）
```cpp
enum class MxScaleAlg : uint8_t { OCP = 0, CUBLAS = 1, DYN_DTYPE_RANGE = 2 };

struct MxQuantConfig {
    MxScaleAlg alg{MxScaleAlg::OCP};
    uint16_t fpEmax{0};             // E4M3:0x0400  E5M2:0x0780  E2M1:0x0100  E1M2:0x0000
    float invDstTypeMax{1.0f};      // cuBLAS: 1/dstTypeMax（host 可覆盖）
    uint16_t addValueBits{0x003f};  // Dyn: dstTypeMax==7 ? 0x001f : 0x003f
    bool zeroScaleOnZeroExp{false}; // OCP zeroMask 判据（见下）
};
```
配置由 Block 在 Init 时按输出 dtype 与 host 参数解析为普通成员
（`fpEmax_` / `invDstTypeMax_` / `addValueBits_`），调用点在函数体内组装
`MxQuantConfig`（tile 类型禁止出现在 Block 类模板的成员声明中——host 编译趟
`compute.h` 的 `__NPU_ARCH__` 守卫不生效，成员声明会被立即解析）。

### OCP 与 cuBLAS 算法选择

- `OCP`按组内最大指数生成2的整数次幂scale，不会根据目标FP8最大有限值额外上调
  scale。归一化结果位于FP8表示边界外时，由后续FP8转换处理。
- `CUBLAS`按组内最大绝对值与目标FP8最大有限值计算scale，并将E8M0指数向上取整，
  可降低边界值量化溢出的风险。
- 两者都按32个元素分组。融合Epilogue只决定量化前的数据来源，不改变这两种算法的
  分组、scale生成和转换语义；因此它们与独立DynamicMxQuant、SwigluMxQuant的同名算法一致。

### OCP zeroMask 语义（zeroScaleOnZeroExp，设计文档 D1）
- **false（gelu_tanh 阵营）**：zeroMask 比较 `sharedExp != 0`，仅作用于 reciprocal
- **true（gelu_mx / swiglu 阵营）**：zeroMask 比较原始 `maxExp != 0`，同时作用于
  yScale 与 reciprocal

两种配置仅在 `0 < maxExp <= fpEmax` 的组上有可观测差异（reciprocal 0 vs 0x7f00）。
调用方应沿用所属Epilogue的既有配置；DynDtypeRange忽略该参数，其实现比较
`sharedExp`且只作用于reciprocal。

### 张量契约（static_assert 门禁）
- **元素类型**：src 为 `bfloat16_t`；maxExp / reciprocal 为 `uint16_t`；
  yScale / y / 转置输入输出为 `int8_t`
- **内存位置**：所有张量必须为 UB
- **排布**：所有张量必须为 `nd_ext_layout_ptn`（仅支持 ND）
- **线性扫描契约**：所有方法按 totalCount / totalScale 连续线性扫描，
  行距由 Block 侧 buffer 布局保证（激活区 rowPitch = Align32(n)，与
  [Gelu](./gelu.md) 输出契约天然衔接；swiglu 为 Align64(n)）
- **maxExp 每组一条**：`totalScale = CeilDiv(totalCount, 32)`；totalCount 可为
  非 32 倍数（如 flat 路径 totalCount = m×n），尾组确定性由调用方保证
  （flat 经 ClearTailVf 对 maxExp 尾部预清零）
- **yScale 打包**：`DIST_PACK_B16`（每寄存器 vlHalf/2 条），调用方 buffer 按
  totalScale 字节预留
- **y 打包**：`DIST_PACK4_B32`（每块 64 元素），FP4 时 N 尾部脏数据由调用方
  负责清零（沿用 gelu_mx 现状契约）
- **TransScaleLayout**：src 为每行 scaleBlockN（ceil(N/32)，可含保留槽）连续
  有效 scale，dst 为 mSize × 32B 行块；dst 前缀之外的陈旧字节可观测时由调用方
  预先清零

## 特殊类型

### MxQuant
```cpp
template <typename DataTypeOut_>
class MxQuant {
public:
    template <typename SrcTensor, typename MaxExpTensor>
    __aicore__ inline void GroupMaxExp(const SrcTensor& srcTensor, const MaxExpTensor& maxExpTensor,
                                         uint32_t totalCount, bool useAbs);

    template <typename MaxExpTensor, typename ScaleTensor, typename ReciprocalTensor>
    __aicore__ inline void GenScale(const MaxExpTensor& maxExpTensor, const ScaleTensor& yScaleTensor,
                                        const ReciprocalTensor& reciprocalTensor, const MxQuantConfig& cfg,
                                        uint32_t totalScale);

    template <typename SrcTensor, typename ReciprocalTensor, typename YTensor>
    __aicore__ inline void Quantize(const SrcTensor& srcTensor, const ReciprocalTensor& reciprocalTensor,
                                    const YTensor& yTensor, uint32_t totalCount,
                                    MxQuantFp4RoundMode fp4RoundMode = MxQuantFp4RoundMode::RINT);

    template <typename ScaleTensor, typename ScaleBlockTensor>
    __aicore__ inline void TransScaleLayout(const ScaleTensor& srcTensor, const ScaleBlockTensor& dstTensor,
                                            uint16_t mSize, uint16_t scaleBlockN);

    template <typename YTensor, typename YBlockTensor>
    __aicore__ inline void TransFp4OutLayout(const YTensor& srcTensor, const YBlockTensor& dstTensor,
                                             uint16_t mSize, uint16_t nSize);
};
```

### MxQuant::GroupMaxExp
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，bfloat16_t，连续 totalCount 元素 |
| maxExpTensor | MaxExpTensor | 输出 UB Tensor，uint16_t，totalScale = CeilDiv(totalCount, 32) 组 |
| totalCount | uint32_t | 连续元素总数（可为非 32 倍数，尾组确定性由调用方保证） |
| useAbs | bool | false → 指数位掩码（OCP）；true → 绝对值掩码（cuBLAS/Dyn） |

### MxQuant::GenScale
| 参数 | 类型 | 说明 |
|------|------|------|
| maxExpTensor | MaxExpTensor | 输入 UB Tensor，uint16_t |
| yScaleTensor | ScaleTensor | 输出 UB Tensor，int8_t（E8M0 打包写入） |
| reciprocalTensor | ReciprocalTensor | 输出 UB Tensor，uint16_t |
| cfg | MxQuantConfig | 算法配置（alg 三选一分派） |
| totalScale | uint32_t | 组数（= CeilDiv(totalCount, 32)） |

### MxQuant::Quantize
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，bfloat16_t，连续 totalCount 元素 |
| reciprocalTensor | ReciprocalTensor | 输入 UB Tensor，uint16_t（GenScale 输出） |
| yTensor | YTensor | 输出 UB Tensor，int8_t（DataTypeOut 打包写入） |
| totalCount | uint32_t | 连续元素总数 |
| fp4RoundMode | MxQuantFp4RoundMode | FP4 舍入（RINT/FLOOR/ROUND，FP8 忽略） |

### MxQuant::TransScaleLayout / TransFp4OutLayout
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，int8_t |
| dstTensor | DstTensor | 输出 UB Tensor，int8_t |
| mSize | uint16_t | 行数 |
| scaleBlockN | uint16_t | 每行有效 scale 数（TransScaleLayout） |
| nSize | uint16_t | 每行元素数（TransFp4OutLayout，行距内部推导） |

## 使用示例

```cpp
#include "blaze/epilogue/tile/compute.h"

// Block 层持有 UB 字节偏移（geluResUbOffset_ / maxExpUbOffset_ ...）
const uint32_t alignedN = Gemm::CeilAlign(nSize, AscendC::ONE_BLK_SIZE);
const uint32_t totalData = mSize * alignedN;
const uint32_t totalScale = Gemm::CeilDiv(totalData, AscendC::ONE_BLK_SIZE);

auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, totalData, totalData);
auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, totalScale, totalScale);
auto actTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(geluResUbOffset_), dataLayout);
auto maxExpTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(maxExpUbOffset_), scaleLayout);
// ... yScaleTensor / reciprocalTensor / yTensor 同理

MxQuantConfig cfg{};              // 函数体内组装（禁止作为 Block 类成员）
cfg.alg = MxScaleAlg::OCP;
cfg.fpEmax = fpEmax_;
cfg.zeroScaleOnZeroExp = false;   // gelu_tanh 阵营；gelu_mx/swiglu 为 true

Blaze::Epilogue::Tile::MxQuant<DataTypeOut> mx;
mx.GroupMaxExp(actTensor, maxExpTensor, totalData, false);
mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);
mx.Quantize(actTensor, reciprocalTensor, yTensor, totalData);
mx.TransScaleLayout(yScaleTensor, yScaleBlockTensor, mSize, scaleBlockN);
```

## 数据流

```
激活结果（bf16, UB, totalCount = mSize × Align32(n)）
    │
    ↓ GroupMaxExp（每 32 元素组取最大指数位/绝对值）
maxExp（uint16, totalScale = CeilDiv(totalCount, 32) 组）
    │
    ↓ GenScale（OCP / cuBLAS / DynDtypeRange 三算法之一）
yScale（E8M0, DIST_PACK_B16 打包）+ reciprocal（uint16, 2^-sharedExp 半精度码点）
    │                                   │
    ↓ Quantize（×reciprocal → SAT cast） │
y（FP8/FP4, DIST_PACK4_B32 打包）        ↓ TransScaleLayout（ceil(N/32) 有效值
                                         │   → 每行 32B 行块布局）
                                         yScale（GM 搬运布局）
```

## 与 Block 层的关系

`MxQuant` 是多阶段计算链 Tile（运行时算法分派 + FP4 舍入运行时分派 +
loop 计数内部推导），当前被四个 Block 直接复用：

- [BlockEpilogueGeluTanhMxQuant](../../block/block_epilogue_gelu_tanh_mx_quant.md)：
  三算法全支持，`zeroScaleOnZeroExp = false`
- [BlockEpilogueGeluMxQuant](../../block/block_epilogue_gelu_mx_quant.md)：
  三算法 + FP4 输出转置（`TransFp4OutLayout`），`zeroScaleOnZeroExp = true`
- [BlockEpilogueFlatQuant](../../block/block_epilogue_flat_quant.md)：
  三算法全支持（dstTypeMax 0/6,7/其他 分派），`zeroScaleOnZeroExp = true`，
  eMax 的 abs/exp 路径由 dstTypeMax ∈ [6,12] 区间判定（保留 flat 原始行为）

- [BlockEpilogueSwigluMxQuant](../../block/block_epilogue_swiglu_mx_quant.md)：
  复用 `GroupMaxExp`、`GenScale`、`Quantize` 和 `TransScaleLayout`，
  `zeroScaleOnZeroExp = true`。该共享 Epilogue 兼容 V2 mode 0 与 V3 mode 2，
  只选择 OCP/cuBLAS，没有动态 dtype range 分支；V3 只公开 `scaleAlg=0/1`，
  `dstTypeMax` 固定为 0。

Block 层保留：fpEmax / invDstTypeMax / addValueBits 的 Init 推导（依赖 host
参数）、UB 布局与偏移、slot/ping-pong 与跨核同步、GM copy、FP4 脏数据清零。
