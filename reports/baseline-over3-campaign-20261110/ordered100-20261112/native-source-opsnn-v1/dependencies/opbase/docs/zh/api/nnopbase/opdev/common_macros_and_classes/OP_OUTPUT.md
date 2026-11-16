# OP\_OUTPUT

## 宏功能

用于封装算子的输出aclTensor和aclTensorList。

## 宏原型

```cpp
OP_OUTPUT(x...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x... | 输入 | 算子输出的aclTensor和aclTensorList。 |

## 约束说明

如果算子包含非aclTensor且非aclTensorList的输出参数，需要先调用aclOpExecutor::ConvertToTensor将参数转换为aclTensor。

## 调用示例

```cpp
// 封装算子的1个输出参数，addOut
OP_OUTPUT(addOut);
```
