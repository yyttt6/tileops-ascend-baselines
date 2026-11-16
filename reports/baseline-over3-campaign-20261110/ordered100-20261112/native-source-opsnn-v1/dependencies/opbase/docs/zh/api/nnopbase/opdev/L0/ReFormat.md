# ReFormat

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

该函数不改变tensor数据的值，在指定format和输入x的维度相同时，将输入数据格式设置为目标format。

具体功能是将用户输入tensor的viewFormat、originalFormat、storageFormat统一为指定的format。

## 函数原型

```cpp
const aclTensor *ReFormat(const aclTensor *x, const op::Format &format, aclOpExecutor *executor=nullptr)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 需要被转换的tensor。数据类型支持FLOAT16、FLOAT32、INT32、UINT32、INT8、UINT8。 |
| format | 输入 | 输入tensor要转换的目标format。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

## 返回值说明

返回数据格式为目标format后的tensor。

## 约束说明

输入tensor的数据的维度需要与指定format的维度相同。

## 调用示例

```cpp
// 将输入reformat成NCHW格式
auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
CHECK_RET(reformatInput != nullptr, nullptr);
```
