# aclDestroyScalarList

## Function

Destroys the aclScalarList created by calling the [aclCreateScalarList](aclCreateScalarList.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyScalarList(const aclScalarList *array)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| aclScalarList to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

The aclScalar in the aclScalarList does not need to be destroyed again.

## Examples

For details, see the [aclCreateScalarList](aclCreateScalarList.md) API call example.
