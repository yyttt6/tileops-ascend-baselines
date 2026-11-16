# SetFloatData

## Function

Initializes the data of a host-side tensor, allocated by `AllocHostTensor`, using a block of memory of type `float`.

## Prototype

```cpp
void SetFloatData(const float *value, uint64_t size, op::DataType dataType)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Pointer to the data memory to be written to `aclTensor`.|
| size | Input| Number of elements to be written.|
| dataType | Input| The data type is `op::DataType` (that is, `ge::DataType`). The value is converted to the specified data type and then written to `aclTensor`.|

## Returns

None

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Initialize a float memory block and assign the values to the first 10 elements of the input.
void Func(const aclTensor *input) {
    float myArray[10];
    input->SetFloatData(myArray, 10, DT_FLOAT);
}
```
