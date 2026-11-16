# ToOpFormat

## Function

Converts `aclFormat` to `op::Format`.

## Prototype

```cpp
Format ToOpFormat(aclFormat format)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| format | Input| Original data format `aclFormat` to be converted.|

## Returns

`op::Format` type

## Restrictions

None

## Examples

```cpp
// Obtain the Format enumeration corresponding to ACL_FORMAT_ND.
void Func() {
    Format format = ToOpFormat(ACL_FORMAT_ND);
}
```
