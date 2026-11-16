# aclCreateScalar

## Function

Creates an aclScalar object as the input parameter for single-operator API execution.

aclScalar is a framework-defined structure used to manage and store scalar data. You can use it without learning about its internal implementation.

## Prototype

```cpp
aclScalar *aclCreateScalar(void *value, aclDataType dataType)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Scalar pointer on the host. Its value will be used as the scalar.|
| dataType | Input| Data type of a scalar.|

## Returns

Created aclScalar on success; else, nullptr.

## Restrictions

- This API must be used together with [aclDestroyScalar](aclDestroyScalar.md). They are used to create and destroy the aclScalar, respectively.
- To create multiple aclScalar objects, call [aclCreateScalarList](aclCreateScalarList.md) to store the scalar list.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclScalar.
float alphaValue = 1.2f;
aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
...
// Use the aclScalar as the input parameter for single-operator API execution.
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, alpha, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// Destroy the aclScalar.
ret = aclDestroyScalar(alpha);
```
