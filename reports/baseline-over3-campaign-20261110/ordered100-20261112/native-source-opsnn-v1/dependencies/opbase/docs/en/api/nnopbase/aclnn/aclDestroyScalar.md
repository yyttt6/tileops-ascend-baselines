# aclDestroyScalar

## Function

Destroys the aclScalar created by calling the [aclCreateScalar](aclCreateScalar.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyScalar(const aclScalar *scalar)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| scalar | Input| Pointer to the scalar to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

None

## Examples

For details, see the [aclCreateScalar](aclCreateScalar.md) API call example.
