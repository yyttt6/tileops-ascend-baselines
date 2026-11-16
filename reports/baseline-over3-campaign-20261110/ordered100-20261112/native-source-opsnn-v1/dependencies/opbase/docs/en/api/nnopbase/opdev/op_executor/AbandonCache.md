# AbandonCache

## Function

Disables the ACLNN cache.

After the first call to an ACLNN operator, the framework records cache information. On subsequent calls, it skips the initial `aclnnXxxGetWorkspaceSize` phase, thereby achieving performance optimization.

## Prototype

```cpp
void AbandonCache(bool disableRepeat = false)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| disableRepeat | Input| Whether `aclOpExecutor` can be reused. The default value is `false`, indicating that it is not reused.|

## Returns

None

## Restrictions

If the `aclnnXxxGetWorkspaceSize` function must be executed in the initial phase, `AbandonCache` needs to be explicitly called in this function.

## Examples

```cpp
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    uniqueExecutor.get()->AbandonCache();
}
```
