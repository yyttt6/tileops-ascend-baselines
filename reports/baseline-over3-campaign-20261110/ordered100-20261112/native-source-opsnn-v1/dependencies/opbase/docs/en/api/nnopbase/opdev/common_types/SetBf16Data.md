# SetBf16Data

## Function

Initializes the tensor data by using a memory block of the bfloat16 type for the host tensor allocated by `AllocHostTensor`.

## Prototype

```cpp
void SetBf16Data(const op::bfloat16 *value, uint64_t size, op::DataType dataType)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Data memory pointer to be written to the aclTensor.|
| size | Input| Number of elements to be written.|
| dataType | Input| The data type is `op::DataType` (`ge::DataType`), which converts data to the specified data type before it is written to the aclTensor.|

## Returns

None

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Initialize a bf16 memory block and assign the value to the first 10 elements of the input.
void Func(const aclTensor *input) {
    bfloat16 myArray[10];
    input->SetBf16Data(myArray, 10, DT_BF16);
}
```
