# aclDestroyBoolArray

## Function

Destroys the aclBoolArray created by calling the [aclCreateBoolArray](aclCreateBoolArray.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyBoolArray(const aclBoolArray *array)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| aclBoolArray to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

None

## Examples

For details, see the [aclCreateBoolArray](aclCreateBoolArray.md) API call example.
