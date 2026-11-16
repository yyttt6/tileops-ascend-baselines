# AllocTensor

## 功能说明

申请一个device侧tensor，提供多个重载函数，可以指定不同的属性。

## 函数原型

```cpp
aclTensor* AllocTensor(const op::Shape& shape, op::DataType dataType, op::Format format = op::Format::FORMAT_ND)
```

```cpp
aclTensor* AllocTensor(const op::Shape& storageShape, const op::Shape& originShape, op::DataType dataType,
                       op::Format storageFormat, op::Format originFormat)
```

```cpp
aclTensor* AllocTensor(op::DataType dataType, op::Format storageFormat, op::Format originFormat)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 将aclTensor的[StorageShape](../common_types/GetStorageShape.md)和[OriginShape](../common_types/GetOriginalShape.md)都设置为指定的shape。 |
| dataType | 输入 | 指定aclTensor的数据类型。 |
| format | 输入 | 将aclTensor的[StorageFormat](../common_types/GetStorageFormat.md)和[OriginFormat](../common_types/GetOriginalFormat.md)都设置为指定的format。 |
| storageShape | 输入 | 将aclTensor的[StorageShape](../common_types/GetStorageShape.md)设置为指定shape。 |
| originShape | 输入 | 将aclTensor的[OriginShape](../common_types/GetOriginalShape.md)设置为指定shape。 |
| storageFormat | 输入 | 将aclTensor的[StorageFormat](../common_types/GetStorageFormat.md)设置为指定format。 |
| originFormat | 输入 | 将aclTensor的[OriginFormat](../common_types/GetOriginalFormat.md)设置为指定format。 |

## 返回值说明

返回申请得到的aclTensor，申请失败则返回nullptr。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 申请一个int64类型，shape为[1, 2, 3, 4, 5]的ND tensor
void Func(aclOpExecutor *executor) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    aclTensor *tensor = executor->AllocTensor(newShape, DT_INT64, ge::FORMAT_ND);
}
```
