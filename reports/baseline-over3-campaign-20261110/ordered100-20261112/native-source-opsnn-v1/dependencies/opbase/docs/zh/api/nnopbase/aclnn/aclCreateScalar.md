# aclCreateScalar

## 功能说明

创建aclScalar对象，作为单算子API执行接口的入参。

aclScalar是框架定义的一种用来管理和存储标量数据的结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```cpp
aclScalar *aclCreateScalar(void *value, aclDataType dataType)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | Host侧的scalar类型的指针，其指向的值会作为scalar。 |
| dataType | 输入 | scalar的数据类型。 |

## 返回值说明

成功则返回创建好的aclScalar，否则返回nullptr。

## 约束说明

- 本接口需与[aclDestroyScalar](aclDestroyScalar.md)接口配套使用，分别完成aclScalar的创建与销毁。
- 如需创建多个aclScalar对象，可调用[aclCreateScalarList](aclCreateScalarList.md)接口来存储标量列表。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclScalar
float alphaValue = 1.2f;
aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
...
// aclScalar作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, alpha, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclScalar
ret = aclDestroyScalar(alpha);
```
