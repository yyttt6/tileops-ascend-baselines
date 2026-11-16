# AbandonCache

## 功能说明

关闭aclnn cache缓存功能。

该功能是指首次调用aclnn算子后框架会记录cache缓存，后续再次调用时，将跳过一阶段aclnnXxxGetWorkspaceSize函数，从而达到性能优化的目的。

## 函数原型

```cpp
void AbandonCache(bool disableRepeat = false)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| disableRepeat | 输入 | 设置aclOpExecutor是否支持复用。默认为false，不复用。 |

## 返回值说明

无

## 约束说明

对于一阶段aclnnXxxGetWorkspaceSize函数必须执行的场景，需在aclnnXxxGetWorkspaceSize函数中显式调用AbandonCache。

## 调用示例

```cpp
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    uniqueExecutor.get()->AbandonCache();
}
```
