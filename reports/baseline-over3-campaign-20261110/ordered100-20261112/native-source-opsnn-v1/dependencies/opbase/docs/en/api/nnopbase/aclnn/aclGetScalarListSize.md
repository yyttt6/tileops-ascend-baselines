# aclGetScalarListSize

## Function

Obtains the size of aclScalarList created by calling the [aclCreateScalarList](aclCreateScalarList.md) API.

## Prototype

```cpp
aclnnStatus aclGetScalarListSize(const aclScalarList *scalarList, uint64_t *size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| scalarList | Input| Input aclScalarList.|
| size | Output| Size of the aclScalarList.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `scalarList` or `size` is a null pointer.

## Restrictions

None

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclScalarList.
std::vector<aclScalar *> tempscalar{alpha1, alpha2};
aclScalarList *scalarList = aclCreateScalarList(tempscalar.data(), tempscalar.size());
...
// Obtain the size of the scalarList.
uint64_t size = 0;
auto ret = aclGetScalarListSize(scalarList, &size); // The obtained size of the scalarList is 2.
...
// Destroy the aclScalarList.
ret = aclDestroyScalarList(scalarList);
```
