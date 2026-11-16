# FloorDiv

## 功能说明

向下取整的除法。

## 函数原型

```cpp
template <typename T>
auto FloorDiv(T x, T y) -> typename std::enable_if<std::is_integral<T>::value, T>::type
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 被除数。 |
| y | 输入 | 除数。 |

## 返回值说明

T：返回向下取整的除法结果，当除数为0时，返回被除数。

## 约束说明

除数和被除数请使用相同类型整型（不包括bool类型）。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
FloorDiv<int32_t>(10, 3)
```
