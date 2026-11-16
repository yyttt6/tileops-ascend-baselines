# OP\_WORKSPACE

## 宏功能

用于封装算子显式指定的workspace参数。

## 宏原型

```cpp
OP_WORKSPACE(x...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x... | 输入 | 算子的workspace参数，表示算子计算过程中需要的Device内存。支持aclTensor或aclTensorList类型。 |

## 约束说明

无。

## 调用示例

```cpp
// 封装算子的workspace参数a
OP_WORKSPACE(a);
```
