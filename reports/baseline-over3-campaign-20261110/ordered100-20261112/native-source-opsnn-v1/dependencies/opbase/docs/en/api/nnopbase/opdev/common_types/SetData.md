# SetData

## Function

Sets the data at a specified position in a host-side tensor allocated by `AllocHostTensor`.

## Prototype

- Set the value at a specified index for a tensor.

    ```cpp
    void SetData(int64_t index, const T value, op::DataType dataType)
    ```

- Initialize tensor data using an existing memory block.

    ```cpp
    void SetData(const T *value, uint64_t size, op::DataType dataType)
    ```

## Parameters

> **Note**: For details about `ge::DataType`, see "ge Namespace > DataType" in [Basic Data Structure and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

- API for setting the value at a specified index for a tensor

    | Parameter| Input/Output| Description|
| --- | --- | --- |
| index | Input| Index of the `aclTensor` element to be set.|
| value | Input| Target value of the specified `aclTensor` element.|
| dataType | Input| The data type is `op::DataType` (that is, `ge::DataType`). The value is converted to the specified data type and then written to `aclTensor`.|

- API for using an existing memory block to initialize tensor data

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
// Initialize an int64_t memory block, set the first 10 elements of the input to the content of the memory, and set the 11th element of the input to the first value of myArray.
void Func(const aclTensor *input) {
    int64_t myArray[10];
    input->SetData(myArray, 10, DT_INT64);
    input->SetData(10, myArray[0], DT_INT64);
}
```
