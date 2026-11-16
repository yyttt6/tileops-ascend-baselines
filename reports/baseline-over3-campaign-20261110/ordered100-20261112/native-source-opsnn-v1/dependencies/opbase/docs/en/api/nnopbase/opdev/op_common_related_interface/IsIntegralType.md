# IsIntegralType

## Function

Checks whether the input data is of integer type, including Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, and Uint64.

## Prototype

```cpp
bool IsIntegralType(const ge::DataType type)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| type | Input| Input data type.|

## Returns

true: integer; false: otherwise.

## Restrictions

None

## Examples

```cpp
// If dtype is not an integer, a return statement is executed.
void Func(const ge::DataType type) {
    if (!IsIntegralType(type)) {
        return;
    }
}
```
