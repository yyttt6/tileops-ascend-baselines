# GetDataType

## Function

Obtains the data type of an aclTensor.

## Prototype

```cpp
op::DataType GetDataType()
```

## Parameters

None

## Returns

`op::DataType` (that is, `ge::DataType`), which indicates the data type of elements in aclTensor, such as float16 or float32.

> **Note**: For details about `ge::DataType`, see "ge Namespace > DataType" in [Basic Data Structures and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

## Restrictions

None

## Examples

```cpp
// Obtain the data type of the input.
void Func(const aclTensor *input) {
    op::DataType dataType = input->GetDataType();
}
```
