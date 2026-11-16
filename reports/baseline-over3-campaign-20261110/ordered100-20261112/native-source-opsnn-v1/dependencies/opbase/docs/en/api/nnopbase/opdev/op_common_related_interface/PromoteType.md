# PromoteType

## Function

Determines the target data type that parameters should be promoted to when performing operations with different data types. For example, when performing an operation between `float16` and `float32`, the `float16` value should first be promoted to `float32`.

## Prototype

```cpp
DataType PromoteType(const DataType type_a, const DataType type_b)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| type_a | Input| Data type of the first parameter.|
| type_b | Input| Data type of the second parameter.|

## Returns

If the promoted data type cannot be determined, the operation fails and DT\_UNDEFINED is returned.

## Restrictions

None

## Examples

```cpp
// Check if dtype can operate with Float32. If not, return.
void Func(const DataType dtype) {
    if (PromoteType(dtype, DT_FLOAT) == DT_UNDEFINED) {
        return;
    }
}
```
