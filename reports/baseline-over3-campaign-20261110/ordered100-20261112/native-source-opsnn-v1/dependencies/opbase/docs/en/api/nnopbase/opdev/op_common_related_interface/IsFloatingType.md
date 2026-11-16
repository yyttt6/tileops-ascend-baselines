# IsFloatingType

## Function

Checks if the input data type is `floating-point`, including `Float64` (`Double`), `Float32` (`Float`), `BFloat16`, and `Float16`.

## Prototype

```cpp
bool IsFloatingType(const DataType dtype)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dtype | Input| Input data type.|

## Returns

If the data type is `floating-point`, **true** is returned. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// If the data type is not floating-point, the function returns.
void Func(const DataType dtype) {
    if (!IsFloatingType(dtype)) {
        return;
    }
}
```
