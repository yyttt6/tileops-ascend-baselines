# GetAicCoreNum

## Function

Obtains the number of AI Cube cores.

## Prototype

```cpp
template <typename T>
uint32_t GetAicCoreNum(const T *context)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| context | Input| Tiling context, which is of type `TilingContext` or `TilingParseContext`.|

## Returns

Number of AI Cube cores, as a `uint32_t`. If **0** is returned, the number failed to be obtained.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
compileInfo->vectorCoreNum = GetAicCoreNum(context_);
```
