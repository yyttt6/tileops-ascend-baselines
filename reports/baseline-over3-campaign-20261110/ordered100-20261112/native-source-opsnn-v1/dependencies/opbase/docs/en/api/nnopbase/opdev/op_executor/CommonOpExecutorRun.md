# CommonOpExecutorRun

## Function

Executes all tasks in the operator executor context based on `workspace` and `stream`.

## Prototype

```cpp
aclnnStatus CommonOpExecutorRun(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| workspace | Input| Pointer to the on-chip memory allocated on the device based on `workspaceSize` computed by the L2 first-phase API.|
| workspaceSize | Input| Workspace size computed by the L2 first-phase API.|
| executor | Input| Operator executor object generated after the L2 first-phase API is called.|
| stream | Input| A specified stream used to execute operator tasks within the executor.|

## Returns

Success: ACLNN\_SUCCESS. Failure: aclnn error code.

## Restrictions

- `executor` must not be empty.
- If `workspaceSize` is greater than 0, `workspace` must not be empty.

## Examples

```cpp
// Second-phase function of aclnnAdd, which is used to execute operators in the executor task queue.
aclnnStatus aclnnAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
```
