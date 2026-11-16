# ToAclDataType

## Function

Converts `op::DataType` to `aclDataType`.

## Prototype

```cpp
aclDataType ToAclDataType(DataType type)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| type | Input| Original data type `op::DataType` to be converted.|

## Returns

`aclDataType` type.

## Restrictions

None

## Examples

```cpp
// Obtain the aclDataType enumeration corresponding to DT_FLOAT.
void Func() {
    aclDataType type = ToAclDataType(DT_FLOAT);
}
```
