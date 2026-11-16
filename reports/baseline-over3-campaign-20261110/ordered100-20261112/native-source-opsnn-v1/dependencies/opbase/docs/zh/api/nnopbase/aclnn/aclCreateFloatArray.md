# aclCreateFloatArray

## 功能说明

创建aclFloatArray对象，作为单算子API执行接口的入参。

aclFloatArray是框架定义的一种用来管理和存储浮点型数据的数组结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```cpp
aclFloatArray *aclCreateFloatArray(const float *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | Host侧的float类型指针，其指向的值会拷贝给aclFloatArray。 |
| size | 输入 | 浮点型数组的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的aclFloatArray，否则返回nullptr。

## 约束说明

- 本接口需与[aclDestroyFloatArray](aclDestroyFloatArray.md)接口配套使用，分别完成aclFloatArray的创建与销毁。
- 调用[aclGetFloatArraySize](aclGetFloatArraySize.md)接口可以获取aclFloatArray的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclFloatArray
std::vector<float> scalesData = {1.0, 1.0, 2.0, 2.0};
aclFloatArray *scales = aclCreateFloatArray(scalesData.data(),scalesData.size());
...
// aclFloatArray作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, scales, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclFloatArray
ret = aclDestroyFloatArray(scales);
```
