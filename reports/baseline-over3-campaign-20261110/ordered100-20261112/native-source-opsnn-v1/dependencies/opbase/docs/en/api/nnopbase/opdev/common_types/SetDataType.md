# SetDataType

## Function

Sets the data type of an aclTensor.

## Prototype

```cpp
void SetDataType(op::DataType dataType)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dataType | Input| The data type is `op::DataType` (`ge::DataType`), which sets the aclTensor to `dataType`.|

## Returns

None

## Restrictions

None

## Examples

```cpp
// Set the data type of the input to int64.
void Func(const aclTensor *input) {
    input->SetDataType(DT_INT64);
}
```
