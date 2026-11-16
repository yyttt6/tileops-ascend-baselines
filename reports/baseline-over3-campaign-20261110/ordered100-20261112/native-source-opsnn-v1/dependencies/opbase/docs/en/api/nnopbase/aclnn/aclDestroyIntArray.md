# aclDestroyIntArray

## Function

Destroys the aclIntArray created by calling the [aclCreateIntArray](aclCreateIntArray.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyIntArray(const aclIntArray *array)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| aclIntArray to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

None

## Examples

For details, see the [aclCreateIntArray](aclCreateIntArray.md) API call example.
