# aclSetOutputTensorAddr

## Function

After aclOpExecutor reuse is enabled by the [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) call, if the output device memory address changes, the device memory address recorded in the output aclTensor needs to be updated.

## Prototype

```cpp
aclnnStatus aclSetOutputTensorAddr(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| executor | Input| aclOpExecutor set to the reusable state.|
| index | Input| Index of the output aclTensor to be updated. The value range is [0, total number of output tensors – 1].|
| tensor | Input| aclTensor pointer to be updated.|
| addr | Input| Device storage address to be updated to the specified aclTensor. The address must be 32-byte aligned. Otherwise, an undefined error may occur.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 561103 is returned, `executor` or `tensor` is a null pointer.
- If error code 161002 is returned, the index value is out of range.
- If error code 161002 is returned, the aclTensor input is `nullptr` when the first-phase API (aclxxXxxGetWorkspaceSize) is called for the first time. The address cannot be updated.

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
aclTensor tensor4= aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor output= aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list[] = {tensor3, tensor4};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
// The AddCustom operator has two inputs (aclTensor) and two outputs (aclTensor and aclTensorList).
// Call the first-phase API.
aclnnAddCustomGetWorkspaceSize(tensor1, tensor2, output, tensorList, &workspaceSize, &executor);
// Set the executor to be reusable.
aclSetAclOpExecutorRepeatable(executor); 
void *addr;
aclSetOutputTensorAddr(executor, 0, output, addr);  // Update the device address of the output aclTensor.
aclSetOutputTensorAddr(executor, 1, output, addr);  // Update the device address of the first aclTensor in the output tensor list.
aclSetOutputTensorAddr(executor, 2, output, addr);  // Update the device address of the second aclTensor in the output tensor list.
...
// Call the second-phase API.
aclnnAddCustom(workspace, workspaceSize, executor, stream);
// Destroy the executor.
aclDestroyAclOpExecutor(executor);  
```
