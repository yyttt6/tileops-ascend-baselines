# aclCreateFloatArray

## Function

Creates an aclFloatArray object as the input parameter for single-operator API execution.

aclFloatArray is a framework-defined array structure used to manage and store floating-point data. You can use it without learning about its internal implementation.

## Prototype

```cpp
aclFloatArray *aclCreateFloatArray(const float *value, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Float pointer on the host. Its value will be copied to the aclFloatArray.|
| size | Input| Length of a floating-point array. The value is a positive integer.|

## Returns

Created aclFloatArray on success; else, nullptr.

## Restrictions

- This API must be used together with [aclDestroyFloatArray](aclDestroyFloatArray.md). They are used to create and destroy the aclFloatArray, respectively.
- You can call the [aclGetFloatArraySize](aclGetFloatArraySize.md) API to obtain the size of the aclFloatArray.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclFloatArray.
std::vector<float> scalesData = {1.0, 1.0, 2.0, 2.0};
aclFloatArray *scales = aclCreateFloatArray(scalesData.data(),scalesData.size());
...
// Use the aclFloatArray as the input parameter for single-operator API execution.
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, scales, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// Destroy the aclFloatArray.
ret = aclDestroyFloatArray(scales);
```
