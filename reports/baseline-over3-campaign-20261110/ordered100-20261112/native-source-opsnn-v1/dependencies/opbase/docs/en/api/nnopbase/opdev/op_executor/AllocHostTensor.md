# AllocHostTensor

## Function

Allocates a tensor on the host. Multiple overloaded functions are provided to specify different input attributes, such as different data types.

## Prototype

- Allocate a tensor on the host based on the input information combination.

    ```cpp
    aclTensor *AllocHostTensor(const op::Shape &shape, op::DataType datatype, op::Format format = op::Format::FORMAT_ND)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const op::Shape &storageShape, const op::Shape &originShape, op::DataType dataType, op::Format storageFormat, op::Format originFormat)
    ```

- Allocate a tensor on the host and use the memory of a specified data type as the tensor content.

    ```cpp
    aclTensor *AllocHostTensor(const int64_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint64_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const bool *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const char *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const int32_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint32_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const int16_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint16_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const int8_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint8_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const double *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const float *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const fp16_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const bfloat16 *value, uint64_t size, op::DataType dataType)
    ```

## Parameters

- Allocate a tensor on the host based on the input information combination.

    | Parameter| Input/Output| Description|
    | --- | --- | --- |
    | shape | Input| Sets [StorageShape](../../opdev/common_types/GetStorageShape.md) and [OriginShape](../../opdev/common_types/GetOriginalShape.md) of aclTensor to the specified shape.|
    | dataType | Input| Specifies the data type of aclTensor.|
    | format | Input| Sets [StorageFormat](../../opdev/common_types/GetStorageFormat.md) and [OriginFormat](../../opdev/common_types/GetOriginalFormat.md) of aclTensor to the specified format.|
    | storageShape | Input| Sets [StorageShape](../../opdev/common_types/GetStorageShape.md) of aclTensor to the specified shape.|
    | originShape | Input| Sets [OriginShape](../../opdev/common_types/GetOriginalShape.md) of aclTensor to the specified shape.|
    | storageFormat | Input| Sets [StorageFormat](../../opdev/common_types/GetStorageFormat.md) of aclTensor to the specified format.|
    | originFormat | Input| Sets [OriginFormat](../../opdev/common_types/GetOriginalFormat.md) of aclTensor to the specified format.|

- Allocate a tensor on the host and use the memory of a specified data type as the tensor content.

    | Parameter| Input/Output| Description|
    | --- | --- | --- |
    | value | Input| Pointer to source data of different data types.|
    | size | Input| Number of elements in the source data.|
    | dataType | Input| Destination data type of the source data, which is written to the tensor.|

## Returns

Success: allocated aclTensor. Failure: nullptr.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Allocate a tensor on the host and copy the data in myArray to the tensor.
void Func(aclOpExecutor *executor) {
    int64_t myArray[10];
    aclTensor *tensor = executor->AllocHostTensor(myArray, 10, DT_INT64);
}
```
