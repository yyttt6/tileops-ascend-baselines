# aclSetAclOpExecutorRepeatable

## Function

Enables aclOpExecutor to be reusable. If you want to reuse existing aclOpExecutor, you must call this API immediately to enable the reuse after the first-phase API aclxxXxxGetworkspaceSize is executed. Later, you can call the second-phase API aclxxXxx multiple times for operator execution.

aclOpExecutor is an operator executor defined by the framework. It serves as a container for executing operator computing. You can directly use it without learning about its internal implementation.

## Prototype

```cpp
aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor *executor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| executor | Input| aclOpExecutor to be reused.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 561103 is returned, `executor` is a null pointer.

## Restrictions

- Currently, operators that use AI CPU and AI Core compute units support aclOpExecutor reuse.
- When a single-operator API is called, aclOpExecutor reuse cannot be enabled in the following scenarios:

  - If L0 APIs related to host-to-device and device-to-device copy are used, such as CopyToNpu, CopyNpuToNpu, and CopyToNpuSync, aclOpExecutor cannot be reused.
  - If the L0 ViewCopy API is used and the source address and destination address of ViewCopy are the same, aclOpExecutor cannot be reused.

- When a single-operator API is called, a device tensor cannot be created in the operator API. Only external tensors can be used.
- aclOpExecutor that is set to the reusable state does not clear the executor resources after the second-phase API is executed. It needs to be used together with [aclDestroyAclOpExecutor](aclDestroyAclOpExecutor.md) to destroy the resources.

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
