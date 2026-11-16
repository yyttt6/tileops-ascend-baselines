# IsRealNumberType

## Function

Checks if the input data type only consists of real-number types.

## Prototype

```cpp
bool IsRealNumberType(const DataType dtype)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dtype | Input| Input data type. It will be checked to see if it only consists of real-number types.<br>The real-number types include `Float64`, `Float32`, `Float16`, `Int16`, `Int32`, `Int64`, `Int8`, `UInt16`, `UInt32`, `UInt64`, `UInt8`, and `BFloat16`.|

## Returns

If the input data type only consists of real-number types, **true** is returned. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// If the data type consists of more than just real-number types, the function returns.
void Func(const DataType dtype) {
    if (!IsRealNumberType(dtype)) {
        return;
    }
}
```
