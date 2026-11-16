# OP\_INPUT

## 宏功能

用于封装算子的输入aclTensor和aclTensorList。

## 宏原型

```cpp
OP_INPUT(x...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x... | 输入 | 输入的aclTensor和aclTensorList。 |

## 约束说明

如果算子包含非aclTensor且非aclTensorList的输入参数，需要先调用aclOpExecutor::ConvertToTensor将参数转换为aclTensor。

## 调用示例

```cpp
// 封装算子的两个输入参数，self和other
OP_INPUT(self, other);
```
