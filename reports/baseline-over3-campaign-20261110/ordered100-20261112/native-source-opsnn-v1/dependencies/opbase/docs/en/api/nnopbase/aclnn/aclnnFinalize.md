# aclnnFinalize

## Function

Deinitializes aclnnXxx in the single-operator API execution framework. Before a process exits, aclnn-related resources in the process must be released. Otherwise, an internal system error occurs, affecting services.

> **Note**:
>Either aclnnFinalize or aclFinalize can be called to deinitialize resources. The difference is that aclnnFinalize releases only aclnn-related resources, while aclFinalize releases aclnn and other resources in the acl API. Therefore, aclnnFinalize is more lightweight than aclFinalize. If both APIs are called, no failure message is returned.

## Prototype

```cpp
aclnnStatus aclnnFinalize()
```

## Parameters

None

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

- This API must be used together with [aclnnInit](aclnnInit.md) to initialize and deinitialize aclnn resources.
- aclnnFinalize can be called only once in a process.

## Examples

For details, see the [aclnnInit](aclnnInit.md) API call example.
