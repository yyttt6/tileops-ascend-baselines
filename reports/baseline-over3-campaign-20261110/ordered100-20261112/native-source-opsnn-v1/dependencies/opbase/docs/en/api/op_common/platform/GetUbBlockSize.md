# GetUbBlockSize

## Function

Obtains the block size of the Unified Buffer (UB) on the platform, in bytes.

## Prototype

```cpp
template <typename T>
uint32_t GetUbBlockSize([[maybe_unused]] const T *context)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| context | Input| Tiling context, which is of type `TilingContext` or `TilingParseContext`.|

## Returns

uint32\_t: block size of the platform UB (32 bytes currently)

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
compileInfo->ubBlockSize = GetUbBlockSize(context_);
```
