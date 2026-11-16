# aclGetRawTensorAddr

## Function

Obtains the device memory address recorded in the aclTensor created by calling the [aclCreateTensor](aclCreateTensor.md) API.

## Prototype

```cpp
aclnnStatus aclGetRawTensorAddr(const aclTensor *tensor, void **addr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| Pointer to the input aclTensor.|
| addr | Output| Device memory address recorded in the aclTensor.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `tensor` or `addr` is a null pointer.

## Restrictions

- This API must be called before the first-phase API aclxxXxxGetWorkspaceSize is called or after the second-phase API aclxxXxx is called. It cannot be called between the first-phase and second-phase API calls.
- This API can be used together with [aclSetRawTensorAddr](aclSetRawTensorAddr.md) to check whether the updated result meets the expectation.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create input and output tensors inputTensor and outputTensor.
std::vector<int64_t> shape = {1, 2, 3};
void *addr1;
void *addr2;
... // Allocate device memory addresses addr1 and addr2.
aclTensor inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr1);
aclTensor outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr2);

void *getAddr1 = nullptr;
void *getAddr2 = nullptr;
// Obtain the device memory address recorded in the inputTensor. The memory address to which the obtained pointer getAddr1 points is the same as addr1.
auto ret = aclGetRawTensorAddr(inputTensor, &getAddr1);
// Obtain the device memory address recorded in the outputTensor. The memory address to which the obtained pointer getAddr2 points is the same as addr2.
auto ret = aclGetRawTensorAddr(outputTensor, &getAddr2);

// Call the first- and second-phase APIs of the Xxx operator.
ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...
```
