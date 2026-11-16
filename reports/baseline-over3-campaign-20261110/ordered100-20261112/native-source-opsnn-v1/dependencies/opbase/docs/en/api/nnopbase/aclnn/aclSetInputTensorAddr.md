# aclSetInputTensorAddr

## Function

After aclOpExecutor reuse is enabled by the [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) call, if the input device memory address changes, the device memory address recorded in the input aclTensor needs to be updated.

## Prototype

```cpp
aclnnStatus aclSetInputTensorAddr(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| executor | Input| aclOpExecutor set to the reusable state.|
| index | Input| Index of the input aclTensor to be updated. The value range is [0, total number of input tensors – 1].|
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
aclSetInputTensorAddr(executor, 0, tensor1, addr);   // Update the device address of the first aclTensor in the input tensor list.
aclSetInputTensorAddr(executor, 1, tensor2, addr);  // Update the device address of the second aclTensor in the input tensor list.
aclSetInputTensorAddr(executor, 2, tensor3, addr);  // Update the device address of the input aclTensor.
...
// Call the second-phase API.
aclnnAddCustom(workspace, workspaceSize, executor, stream);          
// Destroy the executor.
aclDestroyAclOpExecutor(executor);                                  
```
