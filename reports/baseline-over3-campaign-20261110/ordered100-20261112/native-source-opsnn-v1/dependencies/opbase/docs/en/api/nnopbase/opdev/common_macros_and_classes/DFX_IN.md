# DFX\_IN

## Function

Encapsulates the input parameters of the L2 first-phase API aclnn_Xxx_GetWorkspaceSize.

## Prototype

```cpp
DFX_IN(...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| ... | Input| Input parameters of the L2 first-phase API on the host. It is a variable-length parameter.|

## Restrictions

None

## Examples

```cpp
// Input parameters of the add operator. There are three input parameters: self, other, and alpha.
DFX_IN(self, other, alpha);
```
