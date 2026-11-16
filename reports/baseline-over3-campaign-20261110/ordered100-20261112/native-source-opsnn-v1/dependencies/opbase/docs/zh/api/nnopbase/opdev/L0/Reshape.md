# Reshape

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

该函数不改变算子tensor数据，只是将用户传入的输入tensor x的shape转换成该函数的第二个参数shape。

## 函数原型

```cpp
const aclTensor *Reshape(const aclTensor *x, const op::Shape &shape, aclOpExecutor *executor)
```

```cpp
const aclTensor *Reshape(const aclTensor *x, const aclIntArray *shape, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待转换的输入tensor。数据类型和数据格式不限制。输入必须保证是连续内存数据。 |
| shape | 输入 | 转换后的目标shape，支持aclIntArray*、op::Shape（即gert::Shape）类型。数据类型和数据格式不限制。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

## 返回值说明

若Reshape转换成功，则返回带有目标shape信息的aclTensor给调用者；若失败，则返回nullptr。

## 约束说明

- Reshape转换成功的前提是x的ShapeSize需要和第二个参数shape的ShapeSize相等，所谓的ShapeSize举例如下：A的shape=\(1, 3, 256, 256\)，则A的ShapeSize=1\*3\*256\*256。
- 当前不支持转换成空tensor，所谓的空tensor即shape中包含0。

## 调用示例

```cpp
void Func(const aclTensor *x, const op::Shape &shape, aclOpExecutor *executor) {
    auto ret = l0op::Reshape(x, shape, executor);
    return;
}
```
