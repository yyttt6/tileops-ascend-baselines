# GetWorkspaceSize

## 功能说明

根据L2一阶段中调用的L0接口，计算需要的workspace大小。

## 函数原型

```cpp
uint64_t GetWorkspaceSize()
```

## 参数说明

无

## 返回值说明

返回L2接口在运行中需要的workspace大小。

## 约束说明

无

## 调用示例

```cpp
// aclnn固定写法
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
}
```
