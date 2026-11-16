# L0\_DFX

## Function

Collects the delay statistics of the L0 API (Profiling) and prints input parameters.

## Prototype

```cpp
L0_DFX(profilingName, ...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| profilingName | Input| Name of the L0 API on the host, for example, `AddAiCore`.|
| ... | Input| Input parameter of the L0 API on the host. It is a variable-length parameter.|

## Restrictions

1. The L0_DFX macro must be placed in the first line of the L0 function to ensure the accuracy of delay statistics. Incorrect placement can lead to inaccurate statistics or undefined errors.
2. When using this macro in the L0 function, do not enclose the macro with curly brackets. Otherwise, using them can lead to inaccurate delay statistics or undefined errors.
3. Do not call the L0 API that uses the L0_DFX macro recursively or nest it within the function. This prevents conflicts in delay statistics logic.

## Examples

```cpp
// Collect delay statistics of the L0 API AddAiCore and prints parameters. AddAiCore is the name, and self, other, and addOut are the parameters of the L0 API.
L0_DFX(AddAiCore, self, other, addOut);
```
