# aclCreateTensorList

## 功能说明

创建aclTensorList对象，作为单算子API执行接口的入参。

aclTensorList是框架定义的一种用来管理和存储多个张量数据的列表结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```cpp
aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | aclTensor类型的指针数组，其指向的值会赋给TensorList。 |
| size | 输入 | 张量列表的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的aclTensorList，否则返回nullptr。

## 约束说明

- 调用本接口前，需提前调用[aclCreateTensor](aclCreateTensor.md)接口创建aclTensor。
- 本接口需与[aclDestroyTensorList](aclDestroyTensorList.md)接口配套使用，分别完成aclTensorList的创建与销毁。
- 调用[aclGetTensorListSize](aclGetTensorListSize.md)接口可以获取aclTensorList的大小。
- 调用如下接口可刷新输入/输出aclTensorList中记录的Device内存地址。
  - [aclSetDynamicInputTensorAddr](aclSetDynamicInputTensorAddr.md)
  - [aclSetDynamicOutputTensorAddr](aclSetDynamicOutputTensorAddr.md)
  - [aclSetDynamicTensorAddr](aclSetDynamicTensorAddr.md)

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclTensor: input1 input2
std::vector<int64_t> shape = {1, 2, 3};
aclTensor *input1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *input2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
// 创建aclTensorList
std::vector<aclTensor *> tmp{input1, input2};
aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
// aclTensorList作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, tensorList, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclTensorList
ret = aclDestroyTensorList(tensorList);
```
