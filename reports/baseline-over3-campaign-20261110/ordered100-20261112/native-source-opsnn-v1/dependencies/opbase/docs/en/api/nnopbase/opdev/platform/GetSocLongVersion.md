# GetSocLongVersion

## Function

Obtains the long version of the AI processor model, in the format of Ascend_*xxxyy*_.

## Prototype

```cpp
const std::string &GetSocLongVersion()
```

## Parameters

None

## Returns

Long version of the model in string format.

## Restrictions

None

## Examples

```cpp
void func() {
    // Replace Ascendxxx with the actual model version.
    if (GetCurrentPlatformInfo().GetSocLongVersion() != "Ascendxxx") {
        return;
    }
}
```
