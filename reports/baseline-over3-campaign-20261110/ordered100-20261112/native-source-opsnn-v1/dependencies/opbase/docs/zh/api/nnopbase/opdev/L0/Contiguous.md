# Contiguous

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
- <term>Atlas 200I/500 A2 推理产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

该函数为公共的L0接口，作用是将非连续tensor转换为连续tensor。

由于L2级API的输入tensor可能是非连续的，一般L0级算子只支持连续tensor作为输入，因此需要将tensor转为连续后再作为其他L0算子的输入。

输入tensor可以是连续的，接口内部会兼容处理。

## 函数原型

```cpp
const aclTensor *Contiguous(const aclTensor *x, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待转换的输入tensor。数据类型和数据格式不限制。输入不要求是连续内存，但要求所表达的数据在Storage范围内。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

## 返回值说明

若转换成功，则返回一个连续的aclTensor；若失败，则返回nullptr。

## 约束说明

要求输入tensor是合法的tensor，Shape和Stride所表示的数据在Storage大小范围内。例如：shape=\(2, 3\), stride = \(10, 30\), storageSize=8，数据实际空间超过了Storage大小8，该Tensor为非法，Contiguous接口返回nullptr。

## 调用示例

```cpp
// 标准写法，创建OpExecutor
auto uniqueExecutor = CREATE_EXECUTOR();
// self如果非连续，需要转换
auto selfContiguous = l0op::Contiguous(self, executor);
```
