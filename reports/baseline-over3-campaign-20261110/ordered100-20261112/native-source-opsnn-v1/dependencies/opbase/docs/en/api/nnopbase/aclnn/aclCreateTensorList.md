# aclCreateTensorList

## Function

Creates an aclTensorList object as the input parameter for single-operator API execution.

aclTensorList is a framework-defined list structure used to manage and store multiple tensors. You can use it without learning about its internal implementation.

## Prototype

```cpp
aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Pointer array of the aclTensor type. Its value will be assigned to the TensorList.|
| size | Input| Length of the tensor list. The value is a positive integer.|

## Returns

Created aclTensorList on success; else, nullptr.

## Restrictions

- Before calling this API, call [aclCreateTensor](aclCreateTensor.md) to create an aclTensor.
- This API must be used together with [aclDestroyTensorList](aclDestroyTensorList.md). They are used to create and destroy the aclTensorList, respectively.
- You can call the [aclGetTensorListSize](aclGetTensorListSize.md) API to obtain the size of the aclTensorList.
- The following APIs can be called to update the device memory addresses recorded in the input/output aclTensorList:
  - [aclSetDynamicInputTensorAddr](aclSetDynamicInputTensorAddr.md)
  - [aclSetDynamicOutputTensorAddr](aclSetDynamicOutputTensorAddr.md)
  - [aclSetDynamicTensorAddr](aclSetDynamicTensorAddr.md)

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create aclTensors input1 and input2.
std::vector<int64_t> shape = {1, 2, 3};
aclTensor *input1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *input2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
// Create an aclTensorList.
std::vector<aclTensor *> tmp{input1, input2};
aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
// Use the aclTensorList as the input parameter for single-operator API execution.
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, tensorList, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// Destroy the aclTensorList.
ret = aclDestroyTensorList(tensorList);
```
