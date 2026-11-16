# aclDestroyFloatArray

## Function

Destroys the aclFloatArray created by calling the [aclCreateFloatArray](aclCreateFloatArray.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyFloatArray(const aclFloatArray *array)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| aclFloatArray to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

None

## Examples

For details, see the [aclCreateFloatArray](aclCreateFloatArray.md) API call example.
