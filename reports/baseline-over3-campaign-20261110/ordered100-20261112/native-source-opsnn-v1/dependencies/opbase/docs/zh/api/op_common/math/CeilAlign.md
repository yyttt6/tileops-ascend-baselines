# CeilAlign

## 功能说明

以align为单元，向上对齐。

## 函数原型

```Cpp
template <typename T>
auto CeilAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待向上对齐的数值。 |
| align | 输入 | 对齐的单元。 |

## 返回值说明

T：返回以align为单元，向上对齐的结果，当align为0时，返回0。

## 约束说明

待向上对齐的数值和对齐的单元是相同类型整型（不包括bool类型）。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```Cpp
CeilAlign<int32_t>(1000, 64)
```
