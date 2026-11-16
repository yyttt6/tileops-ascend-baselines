# FloorDiv

## Function

Returns the quotient rounded downward to the nearest integer.

## Prototype

```cpp
template <typename T>
auto FloorDiv(T x, T y) -> typename std::enable_if<std::is_integral<T>::value, T>::type
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x | Input| Dividend.|
| y | Input| Divisor.|

## Returns

T: Returns the rounded-down division result. If the divisor is **0**, the dividend is returned.

## Restrictions

The divisor and dividend must be integers of the same type (excluding the `bool` type).

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
FloorDiv<int32_t>(10, 3)
```
