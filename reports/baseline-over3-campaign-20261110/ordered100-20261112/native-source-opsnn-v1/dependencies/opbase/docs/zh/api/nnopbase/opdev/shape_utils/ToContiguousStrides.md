# ToContiguousStrides

## 功能说明

根据已有的shape信息，按照连续规则去推导对应的stride信息。例如shape为\[10, 20, 30\]，那么推导出的stride为\[600, 30, 1\]。

## 函数原型

```cpp
void ToContiguousStrides(const op::Shape &shape, op::Strides &strides)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 记录了一组shape信息，例如一个三维shape：[10, 20, 30]。 |
| strides | 输出 | 记录了一组stride信息，例如针对三维shape的stride：[600, 30, 1]。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 生成一个shape信息为[1, 2, 3, 4, 5]的Shape对象，并生成它的stride数据。
void Func() {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    FVector<int64_t, 25> strides;
    ToContiguousStrides(newShape, strides);
}
```
