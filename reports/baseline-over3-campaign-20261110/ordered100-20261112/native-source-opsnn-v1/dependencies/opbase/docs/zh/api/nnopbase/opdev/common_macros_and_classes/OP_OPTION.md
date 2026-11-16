# OP\_OPTION

## 宏功能

用于封装算子的精度模式。

## 宏原型

```cpp
OP_OPTION(x...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x... | 输入 | 指定算子精度模式，具体取值参见[OpImplMode](OpImplMode.md)。 |

## 约束说明

无

## 调用示例

```cpp
// 封装算子的精度模式为高精度模式
OP_OPTION(IMPL_MODE_HIGH_PRECISION);
```
