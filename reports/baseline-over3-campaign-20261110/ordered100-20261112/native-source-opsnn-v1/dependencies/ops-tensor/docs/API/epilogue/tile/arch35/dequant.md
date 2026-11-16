# Dequant（Tile 级反量化）

> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/dequant.h)

## 功能说明

Tile 级权重反量化组件，把 packed FP4 权重乘以 per-group scale 转成 FP8。对外只有一个
入口：`Dequant<Params>::Run(inputTensor, scaleTensor, outTensor, params)`——Params 类型
选择反量化语义，dtype/layout 差异由 InputTensor 类型编译期分派；VF 内核是类私有静态
成员，不对调用方暴露。

当前提供 per-group W4 特化（group size 32）：一次 VF 调用完成
F4 unpack → F16 → ×scale → F32 → F8 → 交织排布的完整转换，无 temp buffer。

## 特殊约束

### 架构

仅 dav-3510（`__NPU_ARCH__ == 3510`），经
[compute.h](../compute.md) 派发引入；上层组件依赖 `blaze/epilogue/tile/compute.h`，
不直接 include 本头文件。

### 接口形态

函数模板不能偏特化，因此扩展点定义为类模板 `Dequant<Params>`，`Run` 为静态成员函数。
三个 Tensor 是 `Run` 的模板参数，shape、stride、dtype、存储位置和布局优先从 Tensor 类型
及其 layout 取得；Params 只承载 Tensor 无法表达的附加语义，不管理内存：

- dtype/layout 差异 → InputTensor 类型经 `if constexpr` 编译期分派（ND-trans / NZ 两条路径）
- group size、附加句柄等语义 → 具名 Params 类型 + 类模板偏特化
- `DefaultDequantParams`（空 Params）当前无实现，实例化时 static_assert 报错；
  新增常规反量化时优先在默认实现内做 Tensor 分派，存在额外语义时再定义具名 Params

### 数据类型和布局门禁（static_assert）

- 三个 Tensor 均须为 **UB** tensor
- 输出 `fp8_e4m3fn_t`，输入 packed FP4，scale `half` / `bfloat16_t`
- 输入 layout 仅支持 `dn_ext_layout_ptn`（ND-trans 路径）与 `nz_layout_ptn`（NZ 路径）

## 特殊数据结构

### `A8W4TcgDequantParams`

```cpp
template <uint32_t GroupSize_>
struct A8W4TcgDequantParams {
    static constexpr uint32_t GROUP_SIZE = GroupSize_;
    uint32_t validK{0};
    __ubuf__ uint8_t* scaleMaskAddr{nullptr};
};
```


| 字段            | 说明                                                                                                                                                                  |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GROUP_SIZE`    | 编译期 group 大小，当前仅支持 32                                                                                                                                      |
| `validK`        | 运行时 K 有效长度。NZ 路径按`validK / GROUP_SIZE` **floor** 取组数：`k % 32 != 0` 时尾组不转换——这是任何 layout 都无法表达的 floor 语义，故放在 Params              |
| `scaleMaskAddr` | NZ 路径的 VF scale 掩码（一个完整 VF 掩码寄存器，每 64bit 字重复 32bit-on/32bit-off 模式），由调用方的 UB storage 持有；VF 调用边界不保持寄存器，故每次经 Params 重装 |

## 特殊成员方法

### `Dequant<A8W4TcgDequantParams<G>>::Run`

```cpp
template <class InputTensor, class ScaleTensor, class OutTensor>
__aicore__ inline static void Run(const InputTensor& input, const ScaleTensor& scale,
                                  const OutTensor& output, const Params& params);
```

按输入 layout 分派到两条路径，三 Tensor 的布局契约如下：


| 路径     | 输入（packed FP4）                                                     | scale                                                        | 输出（FP8）                                                    |
| :------- | :--------------------------------------------------------------------- | :----------------------------------------------------------- | :------------------------------------------------------------- |
| ND-trans | `dn_ext_layout_ptn` (k, n)，物理 n 行 pitch 64 对齐后 Slice            | `dn_ext_layout_ptn` (kGroup, n)，n 行 pitch 32B 对齐后 Slice | `Weight8BitDnToZnUbLayoutPtn`                                  |
| NZ       | `nz_layout_ptn` (k, n) 显式 stride（紧凑 pitch，n1 间距 = 32 × kLen） | `nd_ext_layout_ptn` (kGroup, n)，行 pitch 为 32 对齐的 N     | `NzRowPaddingLayoutPtn`（256 元素 chunk + BL1 ping-pong 交织） |

scale/输出的布局族见
[layout_struct.h](../../../../../include/blaze/gemm/utils/layout_struct.h)；
两条输出布局同时是 UB→L1 搬运（[CopyPaddedUBToL1](../../../gemm/tile/copy_ub_to_l1.md) 转换权重族）
的源契约，VF 输出布局与 UB→L1 搬运共用同一个 layout 定义（单一事实源）。

## 计算流程

两条路径共用同一条 VF 数学链，差别在加载排布与 store 模式：

1. `DIST_UNPACK4_B8` 从相距 64 字节（128 个 packed FP4 元素）的两列加载并 unpack；
2. `CastWeightF4ToF16`：F4 → F16（scale 为 bf16 时经 bf16 中转再转 F16）；
3. `Mul` per-group scale（ND 路径 scale 经 `Interleave` 交织成两列；NZ 路径从相距
   `BLOCK_CUBE`(16) 元素的两个 scale bank 做相同分布的块加载，用 `scaleMaskAddr` 掩码
   `Select` 逐 lane 选bank）；
4. F16 → F32 → FP8 两轮 `Cast`（RegLayout 0/2 交错），`Select` + `DeInterleave`（ND）
   或 `Select`（NZ）重排 lane；
5. store：ND 路径 `DATA_BLOCK_COPY` 按 `Weight8BitDnToZnUbLayoutPtn` 的 slab 推进；
   NZ 路径 `DIST_PACK_B16` 按 `NzRowPaddingLayoutPtn` 的 chunk/vl/n1 三层 stride 推进。

## 与 Block 层的关系

`Dequant` 只做纯计算：一次 `Run` 处理调用方传入的一个 tile，Tensor 准备、UB slot 生命周期
和多 tile 编排由上层负责。当前由
[Kernel Wqmm Mix Pergroup](../../../gemm/kernel/kernel_wqmm_mix_pergroup.md) 的 AIV prologue
（`KernelPergroupWeightPrologue`）调用：prologue 构造好满足上表契约的 UB tensor view 后
委托本组件，转换结果随后由 UB→L1 搬运读走。

## 使用示例

```cpp
#include "blaze/epilogue/tile/compute.h"

using DequantParams = Blaze::Epilogue::Tile::A8W4TcgDequantParams<32>;
DequantParams params;
params.validK = static_cast<uint32_t>(bubKLen);
params.scaleMaskAddr = scaleMaskUb;   // 调用方 UB storage 持有的掩码

Blaze::Epilogue::Tile::Dequant<DequantParams>::Run(
    weightInUb, scaleUb, weightOutUb, params);
```
