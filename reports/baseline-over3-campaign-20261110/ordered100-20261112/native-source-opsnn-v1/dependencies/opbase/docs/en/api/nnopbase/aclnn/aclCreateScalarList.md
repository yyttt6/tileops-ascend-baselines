# aclCreateScalarList

## Function

Creates an aclScalarList object as the input parameter for single-operator API execution.

aclScalarList is a framework-defined list structure used to manage and store multiple scalars. You can use it without learning about its internal implementation.

## Prototype

```cpp
aclScalarList *aclCreateScalarList(const aclScalar *const *value, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Pointer to the start address of the aclScalar pointer array. The aclScalar pointers in the array are copied to the aclScalarList in sequence.|
| size | Input| Length of the scalar list. The value is a positive integer.|

## Returns

Created aclScalarList on success; else, nullptr.

## Restrictions

- Before calling this API, call [aclCreateScalar](aclCreateScalar.md) to create an aclScalar.
- This API must be used together with [aclDestroyScalarList](aclDestroyScalarList.md). They are used to create and destroy the aclScalarList, respectively.
- You can call the [aclGetScalarListSize](aclGetScalarListSize.md) API to obtain the size of the aclScalarList.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an alpha1 aclScalar.
float alpha1Value = 1.2f;
aclScalar *alpha1 = aclCreateScalar(&alpha1Value, aclDataType::ACL_FLOAT);
// Create an alpha2 aclScalar.
float alpha2Value = 2.2f;
aclScalar *alpha2 = aclCreateScalar(&alpha2Value, aclDataType::ACL_FLOAT);
// Create an aclScalarList.
std::vector<aclScalar *> tempscalar{alpha1, alpha2};
aclScalarList *scalarlist = aclCreateScalarList(tempscalar.data(), tempscalar.size());
...
// Use the aclScalarList as the input parameter for single-operator API execution.
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, scalarlist, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// Destroy the aclScalarList.
ret = aclDestroyScalarList(scalarlist);
```
