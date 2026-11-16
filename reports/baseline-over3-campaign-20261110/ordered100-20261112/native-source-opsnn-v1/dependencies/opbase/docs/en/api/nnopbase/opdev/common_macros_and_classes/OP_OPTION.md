# OP\_OPTION

## Function

Encapsulates the precision mode of an operator.

## Prototype

```cpp
OP_OPTION(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| Precision mode of the operator. For details about the values, see [OpImplMode](OpImplMode.md).|

## Restrictions

None

## Examples

```cpp
// Set the precision mode of the operator to high-precision mode.
OP_OPTION(IMPL_MODE_HIGH_PRECISION);
```
