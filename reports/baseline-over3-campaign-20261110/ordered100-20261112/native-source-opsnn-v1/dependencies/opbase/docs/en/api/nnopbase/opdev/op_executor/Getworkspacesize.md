# GetWorkspaceSize

## Function

Computes the required workspace size based on the L0 API called in the L2 first phase.

## Prototype

```cpp
uint64_t GetWorkspaceSize()
```

## Parameters

None

## Returns

Workspace size required by the L2 API during running.

## Restrictions

None

## Examples

```cpp
// Fixed writing of aclnn
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
}
```
