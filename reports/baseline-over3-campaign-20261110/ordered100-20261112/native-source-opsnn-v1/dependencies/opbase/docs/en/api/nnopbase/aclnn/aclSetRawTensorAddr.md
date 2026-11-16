# aclSetRawTensorAddr

## Function

Updates the device memory address recorded in the aclTensor created by calling the [aclCreateTensor](aclCreateTensor.md) API.

Generally, if the network needs to frequently reuse aclTensor (that is, the attributes such as shape and format remain the same), this API can be used to update the original device memory address of aclTensor.

## Prototype

```cpp
aclnnStatus aclSetRawTensorAddr(aclTensor *tensor, void *addr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| aclTensor pointer to be updated.|
| addr | Input| Device storage address to be updated to the specified aclTensor. The address must be 32-byte aligned. Otherwise, an undefined error may occur.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `tensor` is a null pointer.

## Restrictions

- This API must be called before the first-phase API aclxxXxxGetWorkspaceSize is called or after the second-phase API aclxxXxx is called. It cannot be called between the first-phase and second-phase API calls.
- This API can be used together with [aclGetRawTensorAddr](aclGetRawTensorAddr.md) to check whether the updated result meets the expectation.

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
// Create input and output tensors inputTensor and outputTensor.
std::vector<int64_t> shape = {1, 2, 3};
aclTensor inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
// Call the first- and second-phase APIs of the Xxx operator.
auto ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...

void *addr1;
void *addr2;
... // Allocate device memory addresses addr1 and addr2.
ret = aclSetRawTensorAddr(inputTensor, addr1); // Update the device address of inputTensor.
ret = aclSetRawTensorAddr(outputTensor, addr2); // Update the device address of outputTensor.
...
// After inputTensor and outputTensor are reused, call the first- and second-phase APIs of the Yyy operator in aclnn.
auto ret = aclnnYyyGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclnnYyy(workspace, workspaceSize, executor, stream);
...
```
