# aclCreateBoolArray

## Function

Creates an aclBoolArray object as the input parameter for single-operator API execution.

aclBoolArray is a framework-defined array structure used to manage and store Boolean data. You can use it without learning about its internal implementation.

## Prototype

```cpp
aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Bool pointer on the host. Its value will be copied to the aclBoolArray.|
| size | Input| Length of the Boolean array. The value is a positive integer.|

## Returns

Created aclBoolArray on success; else, nullptr.

## Restrictions

- This API must be used together with [aclDestroyBoolArray](aclDestroyBoolArray.md). They are used to create and destroy the aclBoolArray, respectively.
- You can call [aclGetBoolArraySize](aclGetBoolArraySize.md) to obtain the size of the aclBoolArray.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclBoolArray.
std::vector<bool> maskData = {true, false};
aclBoolArray *mask = aclCreateBoolArray(maskData.data(), maskData.size());
...
// Use the aclBoolArray as the input parameter for single-operator API execution.
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, mask, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// Destroy the aclBoolArray.
ret = aclDestroyBoolArray(mask);
```
