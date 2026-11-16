# CopyToNpu

## 功能说明

创建一个host侧到device侧的数据拷贝任务，并放入executor的任务队列中。

## 函数原型

```cpp
const aclTensor *CopyToNpu(const aclTensor *src, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| src | 输入 | 需要拷贝到device侧的host侧数据。 |
| executor | 输入 | L2接口中一阶段接口声明的算子执行器对象。 |

## 返回值说明

拷贝到device侧后，指向device侧数据的aclTensor。任务创建失败则返回nullptr。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 初始化一个host侧tensor，并拷贝到dst，dst为一个device侧tensor
void Func(aclOpExecutor *executor) {
    int64_t myArray[10];
    auto src = executor->ConvertToTensor(myArray, 10, DT_INT64);
    auto dst = CopyToNpu(src, executor);
}
```
