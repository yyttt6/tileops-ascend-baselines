# SetStorageShape

## 功能说明

设置aclTensor的StorageShape属性。

StorageShape表示aclTensor在内存上的实际排布，即OriginShape实际运行时的shape格式。

## 函数原型

```cpp
void SetStorageShape(const op::Shape &shape)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 数据类型为op::Shape（即gert::Shape），记录了一组shape信息，例如一个三维shape：[10, 20, 30]。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 将input的StorageShape设置为[1, 2, 3, 4, 5]
void Func(aclTensor *input) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    input->SetStorageShape(newShape);
}
```
