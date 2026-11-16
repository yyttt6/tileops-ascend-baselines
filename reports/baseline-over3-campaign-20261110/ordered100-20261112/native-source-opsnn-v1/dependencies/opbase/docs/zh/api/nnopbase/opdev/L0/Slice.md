# Slice

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

从输入tensor中提取所需的切片。

## 函数原型

```cpp
const aclTensor *Slice(const aclTensor *x, const aclTensor *y, const aclTensor *offset, const aclTensor *size, aclOpExecutor *executor)
```

```cpp
const aclTensor *Slice(const aclTensor *x, const aclIntArray *offsets, const aclIntArray *size, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入tensor，数据类型支持FLOAT16、FLOAT、BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、BFLOAT16、UINT64。数据格式支持ND。 |
| y | 输出 | 切片后的输出tensor，数据类型支持FLOAT16、FLOAT、BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、BFLOAT16、UINT64。数据格式支持ND。 |
| offsets | 输入 | const aclIntArray*类型，表示输入x在各个维度切片的起始位置，其形状为x的维度。数据类型支持INT32、INT64。数据格式支持ND。 |
| offset | 输入 | const aclTensor*类型，表示输入x在各个维度切片的起始位置，其形状为x的维度。数据类型支持INT32、INT64。数据格式支持ND。 |
| size | 输入 | 输入x的各个维度切片的大小，其形状为x的维度。支持aclIntArray*、aclTensor*类型。数据类型支持INT32、INT64。数据格式支持ND。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

<!-- npu="A3,910b" id7 -->
> **说明**：BFLOAT16仅适用于如下产品
>
> - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
> - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
<!-- end id7 -->

## 返回值说明

返回类型和输入tensor一样、shape为size的tensor。

## 约束说明

无

## 调用示例

```cpp
// 调用l0op::Slice对每一块进行处理
auto sliceRes = l0op::Slice(self, offsetArray, sizeArray, executor);
// 调用l0op::Slice对每一块进行处理
auto sliceRes = l0op::Slice(xTensor, yTensor, offsetTensor, sizeTensor, executor);
```
