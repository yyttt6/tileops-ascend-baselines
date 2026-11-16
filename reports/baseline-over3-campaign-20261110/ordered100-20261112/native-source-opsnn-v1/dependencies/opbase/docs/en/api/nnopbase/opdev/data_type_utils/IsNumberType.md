# IsNumberType

## Function

Checks if the input data type is numeric.

## Prototype

```cpp
bool IsNumberType(const DataType dtype)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dtype | Input| Input data type. It will be checked to see if it is numeric.<br>The numeric types include `Complex128`, `Complex64`, `Float64`, `Float32`, `Float16`, `Int16`, `Int32`, `Int64`, `Int8`, `QInt32`, `QInt8`, `QUInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt8`, and `BFloat16`.|

## Returns

If the input data type is numeric, **true** is returned. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// If the data type is not numeric, the function returns.
void Func(const DataType dtype) {
    if (!IsNumberType(dtype)) {
        return;
    }
}
```
