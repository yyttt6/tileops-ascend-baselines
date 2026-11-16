# CopyNpuToNpu

## Function

Creates a device-to-device data copy task and enqueues it into the executor's task queue.

## Prototype

```cpp
aclnnStatus CopyNpuToNpu(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| src | Input| Source tensor to be copied.|
| dst | Input| Destination tensor.|
| executor | Input| An operator executor declared in the first phase of an L2 API.|

## Returns

`ACLNN_SUCCESS` (status code **0**) is returned if the copy task is successfully created. Otherwise, and `aclnn` error code is returned.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Create a copy task from src to dst. If the task fails to be created, the function returns.
void Func(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor) {
    if (CopyNpuToNpu(src, dst, executor) != ACLNN_SUCCESS) {
        return;
    }
}
```
