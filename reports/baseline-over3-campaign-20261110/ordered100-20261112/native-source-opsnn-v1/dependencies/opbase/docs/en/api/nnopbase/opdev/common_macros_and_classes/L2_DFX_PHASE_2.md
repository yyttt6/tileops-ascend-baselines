# L2\_DFX\_PHASE\_2

## Function

Collects latency statistics on the L2 second-phase API `aclnn_Xxx_`. It must be called at the beginning of the second-phase API.

## Prototype

```cpp
L2_DFX_PHASE_2(APIName)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| APIName | Input| Name of the L2 API on the host side, for example, aclnn*Xxx*.|

## Restrictions

This macro must be called at the beginning of the L2 second-phase API. Otherwise, the latency statistics may be inaccurate.

## Examples

```cpp
// Collect latency statics on the L2 second-phase API for the abs operator.
L2_DFX_PHASE_2(aclnnAbs);
```
