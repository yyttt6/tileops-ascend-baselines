# aclDestroyAclOpExecutor

## Function

aclOpExecutor is an operator executor defined by the framework. It is a container for executing operator computation. You can use it without learning about its internal implementation.

- For an aclOpExecutor that is not in the reuse state, the framework automatically creates an aclOpExecutor when the first-phase API aclxxXxxGetworkspaceSize is called, and automatically destroys the aclOpExecutor when the second-phase API aclxxXxx is called. You do not need to manually call this API for destroying resources.
- For an aclOpExecutor in the reuse state ([aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) is called to enable reuse), the operator executor is managed by users. As such, this API needs to be explicitly called to destroy the aclOpExecutor.

## Prototype

```cpp
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor *executor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| executor | Input| aclOpExecutor to be destroyed.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 561103 is returned, `executor` is a null pointer.

## Restrictions

This API must be used together with [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md). They are used to reuse and destroy the aclOpExecutor, respectively.

## Examples

For details, see the [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) API call example.
