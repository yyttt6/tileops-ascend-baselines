# aclCreateIntArray

## 功能说明

创建aclIntArray对象，作为单算子API执行接口的入参。

aclCreateIntArray是框架定义的一种用来管理和存储整型数据的数组结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```cpp
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | Host侧的int64_t类型的指针，其指向的值会拷贝给aclIntArray。 |
| size | 输入 | 整型数组的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的aclIntArray，否则返回nullptr。

## 约束说明

- 本接口需与[aclDestroyIntArray](aclDestroyIntArray.md)接口配套使用，分别完成aclIntArray的创建与销毁。
- 调用[aclGetIntArraySize](aclGetIntArraySize.md)接口可以获取aclIntArray的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclIntArray
std::vector<int64_t> sizeData = {1, 1, 2, 3};
aclIntArray *size = aclCreateIntArray(sizeData.data(),sizeData.size());
...
// aclIntArray作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, size, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclIntArray
ret = aclDestroyIntArray(size);
```
