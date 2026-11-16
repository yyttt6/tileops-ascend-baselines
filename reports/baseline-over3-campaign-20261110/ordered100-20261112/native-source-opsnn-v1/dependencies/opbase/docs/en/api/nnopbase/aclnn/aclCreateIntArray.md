# aclCreateIntArray

## Function

Creates an aclIntArray object as the input parameter for single-operator API execution.

aclCreateIntArray is a framework-defined array structure used to manage and store integer data. You can use it without learning about its internal implementation.

## Prototype

```cpp
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Int64_t pointer on the host. Its value will be copied to the aclIntArray.|
| size | Input| Length of an integer array. The value is a positive integer.|

## Returns

Created aclIntArray on success; else, nullptr.

## Restrictions

- This API must be used together with [aclDestroyIntArray](aclDestroyIntArray.md). They are used to create and destroy the aclIntArray, respectively.
- You can call the [aclGetIntArraySize](aclGetIntArraySize.md) API to obtain the size of the aclIntArray.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclIntArray.
std::vector<int64_t> sizeData = {1, 1, 2, 3};
aclIntArray *size = aclCreateIntArray(sizeData.data(),sizeData.size());
...
// Use the aclIntArray as the input parameter for single-operator API execution.
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, size, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// Destroy the aclIntArray.
ret = aclDestroyIntArray(size);
```
