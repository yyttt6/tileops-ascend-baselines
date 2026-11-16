# CeilDiv

## Function

Returns the quotient rounded upward to the nearest integer.

## Prototype

```cpp
template <typename T>
auto CeilDiv(T x, T y) -> typename std::enable_if<std::is_signed<T>::value, T>::type
template <typename T>
auto CeilDiv(T x, T y) -> typename std::enable_if<std::is_unsigned<T>::value, T>::type
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x | Input| Dividend.|
| y | Input| Divisor.|

## Returns

T: Returns the rounded-up division result. If the divisor is **0**, the dividend is returned.

## Restrictions

The divisor and dividend must be integers of the same type (excluding the `bool` type).

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
CeilDiv<int32_t>(5000, 4096)
```
