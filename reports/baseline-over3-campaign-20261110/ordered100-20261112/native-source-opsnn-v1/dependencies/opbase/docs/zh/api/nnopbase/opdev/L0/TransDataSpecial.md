# TransDataSpecial

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

该函数不改变tensor数据的值，实现对tensor数据格式的转换，功能与[TransData](TransData.md)类似。

具体功能是将用户输入tensor的format转换为指定的dstPrimaryFormat。

## 函数原型

```cpp
const aclTensor *TransDataSpecial(const aclTensor *x, op::Format dstPrimaryFormat, int64_t groups, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待转换的tensor。数据类型支持FLOAT16、FLOAT32、INT32、UINT32、INT8、UINT8。 |
| dstPrimaryFormat | 输入 | 输入tensor要转换的目标format。 |
| groups | 输入 | 分组参数，用于分组转换时传入。数据类型支持INT64。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

## 返回值说明

返回数据格式为dstPrimaryFormat的tensor。

## 约束说明

当输入tensor数据类型为FLOAT32、INT32、UINT32时，C0只能按照16处理。

## 调用示例

```cpp
// 标准写法，创建OpExecutor
auto uniqueExecutor = CREATE_EXECUTOR();
// 将gradOutputReFormat的format转换为NC1HWC0
auto gradOutputTransData = l0op::TransDataSpecial(gradOutputReFormat, op::Format::FORMAT_NC1HWC0, 0,uniqueExecutor.get());
```
