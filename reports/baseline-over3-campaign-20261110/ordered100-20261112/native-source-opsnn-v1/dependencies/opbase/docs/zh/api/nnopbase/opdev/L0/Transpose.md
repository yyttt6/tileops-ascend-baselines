# Transpose

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

该函数不改变tensor数据的值，而是对tensor进行转置操作。

通过Transpose算子，可以改变tensor在不同维度上的排列顺序，实现对tensor的维度重排。

具体的是将用户传入的输入tensor x的shape按指定维度的排列顺序perm进行转置并输出。

## 函数原型

- 输入和输出为不同地址

    ```cpp
    const aclTensor *Transpose(const aclTensor *x, const aclTensor *y, const aclTensor *perm, aclOpExecutor *executor)
    ```

- 输入和输出同一地址

    ```cpp
    const aclTensor *Transpose(const aclTensor *x, const aclIntArray *perm, aclOpExecutor *executor)
    ```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入/输出 | 原始输入tensor。输入tensor x需是连续内存数据。数据类型支持FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16。数据格式支持ND。 |
| y | 输出 | 转置后输出tensor。数据类型和数据格式同x。 |
| perm | 输入 | 整型数组，代表输入tensor x的维度，支持aclIntArray*、aclTensor*类型。最多支持8维转置，取值需在[0, x的维度数量-1]范围内。数据类型支持INT32、INT64。数据格式支持ND。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

<!-- npu="A3,910b" id7 -->
> **说明**：BFLOAT16仅适用于如下产品
>
> - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
> - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
<!-- end id7 -->

## 返回值说明

转置成功，返回转置后的tensor；转置失败，返回nullptr。

## 约束说明

- 最多支持8维转置，即输入x和perm的dim至多为8。
- 输入x和perm的dim维度需要一致。

## 调用示例

```cpp
// 标准写法，创建OpExecutor，参数检查
auto uniqueExecutor = CREATE_EXECUTOR();
CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

// 标准写法，将输入self转换成连续的tensor
auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

int64_t dims = selfContiguous->GetViewShape().GetDimNum();
int64_t valuePerm[dims] = {0, 2, 1, 3}; // 表示对原始4维的中间2维做转置，即交换1轴和2轴

auto perm = executor->AllocIntArray(valuePerm, dims);
selfContiguous = l0op::Transpose(selfContiguous, perm, uniqueExecutor.get());
```
