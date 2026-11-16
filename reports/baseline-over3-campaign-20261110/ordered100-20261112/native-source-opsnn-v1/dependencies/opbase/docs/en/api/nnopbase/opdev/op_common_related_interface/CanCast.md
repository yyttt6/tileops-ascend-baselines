# CanCast

## Function

Checks whether a source type can be cast to a destination type.

## Prototype

```cpp
bool CanCast(const DataType from, const DataType to)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| from | Input| Source data type.|
| to | Input| Destination type.|

## Returns

`true`: cast successfully; `false`: otherwise.

## Restrictions

None

## Examples

```cpp
// If dtype cannot be cast to int8, a return statement is executed.
void Func(const DataType dtype) {
    if (!CanCast(dtype, DT_INT8)) {
        return;
    }
}
```
