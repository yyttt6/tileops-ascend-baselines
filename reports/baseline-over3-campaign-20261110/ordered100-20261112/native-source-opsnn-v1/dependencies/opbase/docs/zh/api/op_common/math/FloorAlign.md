# FloorAlign

## 功能说明

以align为单元，向下对齐。

## 函数原型

```cpp
template <typename T>
auto FloorAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待向下对齐的数值。 |
| align | 输入 | 对齐的单元。 |

## 返回值说明

T：返回以align为单元，向下对齐的结果，当align为0时，返回0。

## 约束说明

待向下对齐的数值和对齐的单元是相同类型整型（不包括bool类型）。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
FloorAlign<int32_t>(12345, 4096)
```
