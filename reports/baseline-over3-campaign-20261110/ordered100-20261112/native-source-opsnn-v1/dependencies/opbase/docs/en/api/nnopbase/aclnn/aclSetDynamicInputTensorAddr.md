# aclSetDynamicInputTensorAddr

## Function

After aclOpExecutor reuse is enabled by the [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) call, if the input device memory address changes, the device memory address recorded in the input aclTensorList needs to be updated.

## Prototype

```cpp
aclnnStatus aclSetDynamicInputTensorAddr(aclOpExecutor *executor, size_t irIndex, const size_t relativeIndex, aclTensorList *tensors, void *addr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| executor | Input| aclOpExecutor set to the reusable state.|
| irIndex | Input| Index of aclTensorList to be updated in the operator IR prototype definition, starting from 0.|
| relativeIndex | Input| Index of aclTensor to be updated in aclTensorList. If aclTensorList has N tensors, the value range is [0, N – 1].|
| tensors | Input| aclTensorList pointer to be updated.|
| addr | Input| Device storage address to be updated to the specified aclTensor. The address must be 32-byte aligned. Otherwise, an undefined error may occur.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 561103 is returned, `executor` or `tensors` is a null pointer.
- If error code 161002 is returned, the value of `relativeIndex` is greater than or equal to the number of tensors in `tensors`.
- If error code 161002 is returned, the value of `irIndex` is greater than or equal to the number of input parameters of the operator prototype.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
// Create the input and output (aclTensor and aclTensorList).
std::vector<int64_t> shape = {1, 2, 3};
aclTensor tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor3 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor output = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list[] = {tensor1, tensor2};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
// The AddCustom operator has two inputs (aclTensorList and aclTensor) and one output (aclTensor).
// Call the first-phase API.
aclnnAddCustomGetWorkspaceSize(tensorList, tensor3, output, &workspaceSize, &executor);
// Set the executor to be reusable.
aclSetAclOpExecutorRepeatable(executor);  
void *addr;
aclSetDynamicInputTensorAddr(executor, 0, 0, tensorList, addr);   // Update the device address of the first aclTensor in the input tensor list.
aclSetDynamicInputTensorAddr(executor, 0, 1, tensorList, addr);  // Update the device address of the second aclTensor in the input tensor list.
...
// Call the second-phase API.
aclnnAddCustom(workspace, workspaceSize, executor, stream);
// Destroy the executor.
aclDestroyAclOpExecutor(executor);  
```
