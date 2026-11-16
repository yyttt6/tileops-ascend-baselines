# CopyNpuToNpu

## 功能说明

创建一个device侧到device侧的数据拷贝任务，并放入executor的任务队列中。

## 函数原型

```cpp
aclnnStatus CopyNpuToNpu(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| src | 输入 | 拷贝的源tensor。 |
| dst | 输入 | 拷贝的目的tensor。 |
| executor | 输入 | L2接口中一阶段接口声明的算子执行器对象。 |

## 返回值说明

创建拷贝任务成功，则返回ACLNN\_SUCCESS（状态码值0），否则返回其他aclnn错误码。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 创建src到dst的拷贝任务， 如果不成功则返回
void Func(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor) {
    if (CopyNpuToNpu(src, dst, executor) != ACLNN_SUCCESS) {
        return;
    }
}
```
