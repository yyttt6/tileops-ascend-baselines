# aclDestroyTensor

## Function

Destroys the aclTensor created by calling the [aclCreateTensor](aclCreateTensor.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyTensor(const aclTensor *tensor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| Pointer to the tensor to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

None

## Examples

For details, see the [aclCreateTensor](aclCreateTensor.md) API call example.
