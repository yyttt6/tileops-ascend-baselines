# ReleaseTo

## 功能说明

将UniqueExecutor中保存的aclOpExecutor指针，传递到L2一阶段输出的executor中。

## 函数原型

```cpp
void ReleaseTo(aclOpExecutor **executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| executor | 输出 | 出参，将UniqueExecutor中的aclOpExecutor指针赋值给*executor。 |

## 返回值说明

无

## 约束说明

executor非空。

## 调用示例

```cpp
// aclnn固定写法
void aclnnAddGetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor) {
    auto uniqueExecutor = CREATE_EXECUTOR();
    ......
    uniqueExecutor.ReleaseTo(executor);
}
```
