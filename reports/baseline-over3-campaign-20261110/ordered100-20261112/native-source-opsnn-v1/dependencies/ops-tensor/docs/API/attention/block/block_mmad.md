# Block Mmad 基础框架
> 公共接口说明

## 概述

Attention Block 层矩阵乘计算组件的基础框架。与 Gemm 模块的 `BlockMmad` 类似，通过 SFINAE 对 `DispatchPolicy` 进行特化选择。主模板未匹配时触发编译期 `static_assert` 报错。

详见：[README.md](./README.md) 查看 API 清单和实现对比。

## 类模板概述

### 模板定义

```cpp
template <class DispatchPolicy_, class QType_, class LayoutQ_, class KType_, class LayoutK_, class VType_,
           class LayoutV_, class OutType_, class LayoutOut_, class... ExtArgs_>
class BlockMmad;
```

### 模板参数

| 参数 | 说明 |
|------|------|
| DispatchPolicy_ | 调度策略类型（如 `BlockFlatQuant`） |
| QType_ | Q 矩阵数据类型（如 `bfloat16_t`） |
| LayoutQ_ | Q 矩阵布局类型（如 `nd_ext_layout_ptn`） |
| KType_ | K 矩阵数据类型 |
| LayoutK_ | K 矩阵布局类型 |
| VType_ | V 矩阵数据类型 |
| LayoutV_ | V 矩阵布局类型 |
| OutType_ | Out 矩阵（输出）数据类型 |
| LayoutOut_ | Out 矩阵布局类型 |
| ExtArgs_ | 可变模板参数，供后续 `DispatchPolicy` 特化扩展携带额外类型参数，不影响原有 9 个参数的顺序与语义 |

### SFINAE 分发机制

主模板（未匹配）触发编译期报错：

```cpp
template <class DispatchPolicy_, class QType_, class LayoutQ_, class KType_, class LayoutK_, class VType_,
           class LayoutV_, class OutType_, class LayoutOut_, class... ExtArgs_>
class BlockMmad {
    static_assert(Blaze::Gemm::always_false_v<DispatchPolicy_>,
                  "BlockMmad is not implemented for this DispatchPolicy");
};
```

特化版本通过显式特化 `BlockMmad<BlockFlatQuant<>, ...>` 匹配。

### 特化版本

| DispatchPolicy | 特化文件 | 说明 |
|---------------|---------|------|
| `BlockFlatQuant<>` | [block_mmad_flat_quant.md](./block_mmad_flat_quant.md) | 双矩阵乘 Block（A×P2→L1, P1×temp→UB） |

## 公共约束

1. **模板参数要求**：
   - DispatchPolicy 必须提供 `ScheduleType` 类型，用于 Kernel 层 SFINAE 匹配
   - LayoutQ/LayoutK 必须为 asc::te 的合法布局类型

2. **SFINAE 匹配**：
   - 未匹配的 DispatchPolicy 会触发编译期 `static_assert` 报错
   - 匹配成功后进入对应特化实现

## 公共调用示例

### 组件组装模板

```cpp
using DispatchPolicy = Blaze::Attention::BlockFlatQuant<>;
using BlockMmad = Blaze::Attention::Block::BlockMmad<
    DispatchPolicy, QType, LayoutQ, KType, LayoutK, VType, LayoutV, OutType, LayoutOut>;
```
