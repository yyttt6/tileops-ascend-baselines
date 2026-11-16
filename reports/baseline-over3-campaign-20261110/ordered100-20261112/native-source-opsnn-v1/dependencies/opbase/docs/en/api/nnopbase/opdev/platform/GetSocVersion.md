# GetSocVersion

## Function

Obtains the short version of the AI processor model, in the format of Ascend_*xxx*_.

## Prototype

```cpp
SocVersion GetSocVersion()
```

## Parameters

None

## Returns

`SoCversion`

## Restrictions

None

## Examples

```cpp
void func() {
    // Replace ASCENDXXX with the actual model version.
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCENDXXX) {
        return;
    }
}
```
