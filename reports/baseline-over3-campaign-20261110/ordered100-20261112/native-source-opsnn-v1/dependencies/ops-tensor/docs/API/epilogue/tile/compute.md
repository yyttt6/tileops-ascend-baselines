# Compute（Tile 架构派发入口）
> [代码位置](../../../../include/blaze/epilogue/tile/compute.h)

## 功能说明

`compute.h` 是 Epilogue Tile 计算组件的架构派发入口（Architecture Dispatcher）。
它按 `__NPU_ARCH__` 条件编译引入当前架构对应的 `arch35/` Tile 头文件，
是 Block 层访问 Tile 层的**唯一合法 include 路径**。

```cpp
#pragma once

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#include "blaze/epilogue/tile/arch35/gelu.h"
#include "blaze/epilogue/tile/arch35/initialize_empty_softmax.h"
#include "blaze/epilogue/tile/arch35/reduce_square.h"
#include "blaze/epilogue/tile/arch35/rms_softmax.h"
#endif
```

## 特殊约束

### include 规范
Block Epilogue 只能 `#include "blaze/epilogue/tile/compute.h"`，
**禁止**直接 include `tile/arch35/<name>.h`：
- Tile 实现与具体 NPU 架构绑定，直连会使 Block 头文件失去架构可移植性
- 新架构（如后续 arch 目录）只需扩展 compute.h，Block 层代码零改动

### 注册规范
新增 Tile 时必须在 compute.h 的 `__NPU_ARCH__ == 3510` 守卫内注册，
include 列表保持**字母序**。未注册的 Tile 对 Block 层不可见
（Block 经由 compute.h 引入，直连 arch35 头属于违规）。

### 架构守卫
当前仅注册 `__NPU_ARCH__ == 3510`（Ascend 950 / dav-3510）的实现。
非 3510 编译单元中 Tile 类型不可见，引用 Tile 类的代码只应出现在
架构确定的 Kernel/Block 路径上。

## 使用方式

```cpp
#include "blaze/epilogue/tile/compute.h"

// BlockEpilogue 实现内部
auto layout = Gemm::MakeNDExtLayout(mSize, nSize, nAligned);
auto srcTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, float>(0), layout);
auto dstTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(dstOffset), layout);
Blaze::Epilogue::Tile::Gelu<bfloat16_t, float> gelu;
gelu.GeluTanh(srcTensor, dstTensor, mSize, nSize);
```

## 数据流

```
BlockEpilogue（管理 UB 偏移/同步/slot）
    ↓ #include
tile/compute.h（按 __NPU_ARCH__ 派发）
    ↓
tile/arch35/<name>.h（架构实现，VF 寄存器级计算）
```
