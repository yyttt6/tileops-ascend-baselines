# IsBasicType

## Function

Checks whether the input data type is a basic type.

## Prototype

```cpp
bool IsBasicType(const DataType dtype)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dtype | Input| Input data type, which is used to determine whether the data type is a basic data type.<br>The basic types include Complex128, Complex64, Float64, Float32, Float16, Int16, Int32, Int64, Int8, QInt16, QInt32, QInt8, QUInt16, QUInt8, UInt16, UInt32, UInt64, UInt8, and BFloat16.|

## Returns

If the input data type is a basic type, true is returned. Otherwise, false is returned.

## Restrictions

None

## Examples

```cpp
// If the data type is not a basic type, a return statement is executed.
void Func(const DataType dtype) {
    if (!IsBasicType(dtype)) {
        return;
    }
}
```
