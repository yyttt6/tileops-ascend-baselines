# IsIntegralType (Including Bool)

## Function

Checks if the input data type is `integer`, including `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, and `Uint64`.

If include\_bool is set to `true`, `bool` is also considered an `integer`.

## Prototype

```cpp
bool IsIntegralType(const ge::DataType type, const bool include_bool)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| type | Input| Input data type.|
| include_bool | Input| Whether to consider `bool` an `integer`.|

## Returns

If the data type is `integer`, **true** is returned. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// If the data type is not integer, the function returns. Note that bool is also considered an integer.
void Func(const ge::DataType type) {
    if (!IsIntegralType(type, true)) {
        return;
    }
}
```
