# ToShapeVector

## 功能说明

提取输入Shape对象中的维度信息到一个ShapeVector容器。

## 函数原型

```cpp
FVector<int64_t, 25> ToShapeVector(const op::Shape &shape)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 记录了一组shape信息，例如一个三维shape：[10, 20, 30]。 |

## 返回值说明

返回一个FVector对象，其中存放着每一维的shape大小。

## 约束说明

无

## 调用示例

```cpp
// 生成一个shape信息为[1, 2, 3, 4, 5]的Shape对象，并将它转为FVector。
void Func() {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    auto shapeVec = ToShapeVector(newShape);
}
```
