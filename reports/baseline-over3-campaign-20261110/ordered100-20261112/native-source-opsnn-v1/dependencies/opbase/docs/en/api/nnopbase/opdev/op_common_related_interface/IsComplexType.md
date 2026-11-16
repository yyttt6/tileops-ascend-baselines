# IsComplexType

## Function

Checks if the input data type is a complex type, including Complex128, Complex64, and Complex32.

## Prototype

```cpp
bool IsComplexType(const DataType dtype)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dtype | Input| Input data type.|

## Returns

If the input data type is complex, **true** is returned. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// If the data type is not complex, the function returns.
void Func(const DataType dtype) {
    if (!IsComplexType(dtype)) {
        return;
    }
}
```
