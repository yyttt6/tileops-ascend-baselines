# ToAclFormat

## Function

Converts `op::Format` to `aclFormat`.

## Prototype

```cpp
aclFormat ToAclFormat(Format format)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| format | Input| Original data format `op::Format` to be converted.|

## Returns

`aclFormat` format.

## Restrictions

None

## Examples

```cpp
// Obtain the aclFormat enumeration mapped from FORMAT_ND.
void Func() {
    aclFormat format = ToAclFormat(FORMAT_ND);
}
```
