# GetCurrentPlatformInfo

## Function

Obtains NPU platform information, including the AI processor model and parameters.

## Prototype

```cpp
const PlatformInfo &GetCurrentPlatformInfo()
```

## Parameters

None

## Returns

Platform information.

## Restrictions

None

## Examples

```cpp
// Obtain the platform information.
void func() {
    const PlatformInfo npuInfo = GetCurrentPlatformInfo();
}
```
