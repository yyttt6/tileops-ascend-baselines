# aclCreateBoolArray

## 功能说明

创建aclBoolArray对象，作为单算子API执行接口的入参。

aclBoolArray是框架定义的一种用来管理和存储布尔型数据的数组结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```cpp
aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | Host侧的bool类型指针，其指向的值会拷贝给aclBoolArray。 |
| size | 输入 | 布尔型数组的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的aclBoolArray，否则返回nullptr。

## 约束说明

- 本接口需与[aclDestroyBoolArray](aclDestroyBoolArray.md)接口配套使用，分别完成aclBoolArray的创建与销毁。
- 调用[aclGetBoolArraySize](aclGetBoolArraySize.md)接口可以获取aclBoolArray的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclBoolArray
bool maskData[] = {true, false};
aclBoolArray *mask = aclCreateBoolArray(maskData, sizeof(maskData) / sizeof(maskData[0]));
...
// aclBoolArray作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, mask, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclBoolArray
ret = aclDestroyBoolArray(mask);
```
