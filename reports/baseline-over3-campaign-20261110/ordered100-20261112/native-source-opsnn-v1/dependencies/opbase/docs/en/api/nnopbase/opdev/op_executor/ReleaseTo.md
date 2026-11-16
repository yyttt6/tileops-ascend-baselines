# ReleaseTo

## Function

Passes the `aclOpExecutor` pointer stored in `UniqueExecutor` to the `executor` produced at the first stage of an L2 API.

## Prototype

```cpp
void ReleaseTo(aclOpExecutor **executor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| executor | Output| Output parameter. It assigns the `aclOpExecutor` pointer from `UniqueExecutor` to `executor`.|

## Returns

None

## Restrictions

`executor` cannot be `null`.

## Examples

```cpp
// Fixed format for aclnn
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    uniqueExecutor.ReleaseTo(executor);
}
```
