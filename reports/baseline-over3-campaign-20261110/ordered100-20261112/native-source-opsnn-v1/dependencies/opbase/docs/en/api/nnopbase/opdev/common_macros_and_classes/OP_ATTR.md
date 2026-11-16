# OP\_ATTR

## Function

Encapsulates attribute parameters of an operator.

## Prototype

```cpp
OP_ATTR(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| Operator attribute parameters (attribute parameters in the operator prototype).|

## Restrictions

None

## Examples

```cpp
// Encapsulate two attribute parameters of an operator: dstType and sqrtMode.
OP_ATTR(dstType, sqrtMode);
```
