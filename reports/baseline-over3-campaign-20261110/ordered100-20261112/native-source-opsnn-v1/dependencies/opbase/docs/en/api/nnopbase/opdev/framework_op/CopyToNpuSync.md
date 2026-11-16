# CopyToNpuSync

## Function

Copies data from the host side to the device side synchronously and does not enqueue the task into the executor's task queue. The API blocks until the copy operation is completed. The caller is responsible for freeing the device memory required by this API.

## Prototype

```cpp
const aclTensor *CopyToNpuSync(const aclTensor *src, aclOpExecutor *executor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| src | Input| Data to be copied from the host side to the device side.|
| executor | Input| An operator executor declared in the first phase of an L2 API.|

## Returns

If the task creation succeeds, an `aclTensor` is returned, pointing to the device-side data after the copy. If the task creation fails, `nullptr` is returned.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Initialize a tensor on the host side and copy it to dst, which is a tensor on the device side.
void Func(aclOpExecutor *executor) {
    int64_t myArray[10];
    auto src = executor->ConvertToTensor(myArray, 10, DT_INT64);
    auto dst = CopyToNpuSync(src, executor);
}
```
