# CeilAlign

## Function

Aligns upward to the nearest multiple of *align*.

## Prototype

```Cpp
template <typename T>
auto CeilAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x | Input| Value to be aligned upward.|
| align | Input| Alignment unit.|

## Returns

`T`: Returns the value aligned upward to the nearest multiple of *align*. If *align* is **0**, **0** is returned.

## Restrictions

The value to be aligned and the alignment unit must be integers of the same type (excluding the `bool` type).

## Examples

The following code is for reference only and should not be copied directly for execution:

```Cpp
CeilAlign<int32_t>(1000, 64)
```
