# GetUbSize

## Function

Obtains the size of the Unified Buffer (UB) on the platform, in bytes.

## Prototype

```cpp
template <typename T>
uint32_t GetUbSize(const T *context)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| context | Input| Tiling context, which is of type `TilingContext` or `TilingParseContext`.|

## Returns

Size of the platform UB, as a uint64\_t. If **0** is returned, the size failed to be obtained.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
compileInfo->ubSize = GetUbSize(context_);
```
