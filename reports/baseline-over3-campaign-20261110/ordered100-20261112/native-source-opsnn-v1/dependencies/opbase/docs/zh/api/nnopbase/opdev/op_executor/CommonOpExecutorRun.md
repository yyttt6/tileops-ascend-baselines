# CommonOpExecutorRun

## 功能说明

根据workspace、stream，执行算子executor上下文中的所有任务。

## 函数原型

```cpp
aclnnStatus CommonOpExecutorRun(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| workspace | 输入 | 根据L2一阶段接口计算出的workspaceSize，在device侧申请的片上内存指针。 |
| workspaceSize | 输入 | L2一阶段接口计算出的workspaceSize。 |
| executor | 输入 | L2一阶段接口调用后生成的op执行器对象。 |
| stream | 输入 | 指定流执行executor中的算子任务。 |

## 返回值说明

执行成功返回ACLNN\_SUCCESS，否则返回aclnn错误码。

## 约束说明

- executor不能为空。
- workspaceSize大于0时，workspace不能为空。

## 调用示例

```cpp
// aclnnAdd的二阶段函数，执行executor任务队列中的算子
aclnnStatus aclnnAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
```
