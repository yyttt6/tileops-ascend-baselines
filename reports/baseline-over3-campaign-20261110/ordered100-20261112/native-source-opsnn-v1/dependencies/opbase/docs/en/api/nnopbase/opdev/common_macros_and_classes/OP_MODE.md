# OP\_MODE

## Function

Encapsulates an operator's execution mode, specifically whether the HF32 data type is enabled during computation.

## Prototype

```cpp
OP_MODE(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| Operator's execution mode. For details, see [OpExecMode](OpExecMode.md). The default value is **0**, indicating the `Default` mode.|

## Restrictions

None

## Examples

```cpp
// Encapsulate an operator's execution mode.
OP_MODE(OP_EXEC_MODE_DEFAULT);
```
