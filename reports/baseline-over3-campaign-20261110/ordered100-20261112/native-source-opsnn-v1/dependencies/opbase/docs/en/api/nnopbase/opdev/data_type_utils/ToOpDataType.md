# ToOpDataType

## Function

Converts `aclDataType` to `op::DataType`.

## Prototype

```cpp
DataType ToOpDataType(aclDataType type)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| type | Input| Original data type `aclDataType` to be converted.|

## Returns

`op::DataType` type

## Restrictions

None

## Examples

```cpp
// Obtain the DataType enumeration corresponding to ACL_FLOAT.
void Func() {
    DataType type = ToOpDataType(ACL_FLOAT);
}
```
