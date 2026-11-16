# OP\_WORKSPACE

## Function

Encapsulates the workspace parameter explicitly specified by the operator.

## Prototype

```cpp
OP_WORKSPACE(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| Workspace parameter of the operator, indicating the device memory required for operator computation. The aclTensor or aclTensorList type is supported.|

## Restrictions

None

## Examples

```cpp
// Encapsulate the workspace parameter a of the operator.
OP_WORKSPACE(a);
```
