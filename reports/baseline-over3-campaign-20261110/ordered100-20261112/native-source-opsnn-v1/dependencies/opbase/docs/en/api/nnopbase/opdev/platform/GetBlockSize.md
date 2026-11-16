# GetBlockSize

## Function

Obtains the size (in bytes) of a data block on the current NPU platform.

## Prototype

```cpp
int64_t GetBlockSize()
```

## Parameters

None

## Returns

int64\_t is returned.

## Restrictions

None

## Examples

```cpp
// Obtain the size of a data block on the current NPU platform.
void func() {
    int64_t size = GetCurrentPlatformInfo().GetBlockSize();
}
```
