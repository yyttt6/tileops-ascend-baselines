# AllocTensor

## Function

Allocates a tensor on the device. Multiple overloaded functions are provided to specify different attributes.

## Prototype

```cpp
aclTensor *AllocTensor(const Shape &shape, DataType dataType, Format format = FORMAT_ND)
```

```cpp
aclTensor *AllocTensor(const Shape &storageShape, const Shape &originShape, DataType dataType, Format storageFormat, Format originFormat)
```

```cpp
aclTensor *AllocTensor(DataType dataType, Format storageFormat, Format originFormat)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| Sets [StorageShape](../../opdev/common_types/GetStorageShape.md) and [OriginShape](../../opdev/common_types/GetOriginalShape.md) of aclTensor to the specified shape.|
| dataType | Input| Specifies the data type of aclTensor.|
| format | Input| Sets [StorageFormat](../../opdev/common_types/GetStorageFormat.md) and [OriginFormat](../../opdev/common_types/GetOriginalFormat.md) of aclTensor to the specified format.|
| storageShape | Input| Sets [StorageShape](../../opdev/common_types/GetStorageShape.md) of aclTensor to the specified shape.|
| originShape | Input| Sets [OriginShape](../../opdev/common_types/GetOriginalShape.md) of aclTensor to the specified shape.|
| storageFormat | Input| Sets [StorageFormat](../../opdev/common_types/GetStorageFormat.md) of aclTensor to the specified format.|
| originFormat | Input| Sets [OriginFormat](../../opdev/common_types/GetOriginalFormat.md) of aclTensor to the specified format.|

## Returns

Success: allocated aclTensor. Failure: nullptr.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Allocate an int64 ND tensor with shape [1, 2, 3, 4, 5].
void Func(aclOpExecutor *executor) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    aclTensor *tensor = executor->AllocTensor(newShape, DT_INT64, ge::FORMAT_ND);
}
```
