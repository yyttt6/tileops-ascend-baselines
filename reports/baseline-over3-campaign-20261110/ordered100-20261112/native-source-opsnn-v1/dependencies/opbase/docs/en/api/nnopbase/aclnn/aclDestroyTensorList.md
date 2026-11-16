# aclDestroyTensorList

## Function

Destroys the aclTensorList created by calling the [aclCreateTensorList](aclCreateTensorList.md) API.

## Prototype

```cpp
aclnnStatus aclDestroyTensorList(const aclTensorList *array)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| aclTensorList to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

The aclTensor in the aclTensorList does not need to be destroyed again.

## Examples

For details, see the [aclCreateTensorList](aclCreateTensorList.md) API call example.
