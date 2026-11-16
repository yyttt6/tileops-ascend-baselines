# Cast

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

将输入tensor转换为指定的数据类型。

## 函数原型

```cpp
const aclTensor *Cast(const aclTensor *self, op::DataType dstDtype, aclOpExecutor *executor)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| self | 输入 | 待转换的输入tensor，数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX64、COMPLEX128。数据格式支持ND。 |
| dstDtype | 输入 | 转换后的目标dtype，数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX64、COMPLEX128。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

<!-- npu="A3,910b" id7 -->
> **说明**：BFLOAT16仅适用于如下产品
>
> - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
> - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
<!-- end id7 -->

## 返回值说明

返回类型为dstDtype的tensor。

## 约束说明

无

## 调用示例

```cpp
// 标准写法，创建OpExecutor
auto uniqueExecutor = CREATE_EXECUTOR();

auto selfCasted = self;
// 当self为布尔类型时，利用Cast接口转换为uint8类型后可进行整型计算
if (self->GetDataType() == op::DataType::DT_BOOL) {
  selfCasted = l0op::Cast(self, op::DataType::DT_UINT8, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_PARAM_NULLPTR);
}
```
