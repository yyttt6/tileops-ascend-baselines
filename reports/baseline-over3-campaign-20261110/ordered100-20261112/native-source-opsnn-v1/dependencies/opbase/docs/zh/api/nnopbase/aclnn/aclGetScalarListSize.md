# aclGetScalarListSize

## 功能说明

获取aclScalarList的大小，aclScalarList通过[aclCreateScalarList](aclCreateScalarList.md)接口创建。

## 函数原型

```cpp
aclnnStatus aclGetScalarListSize(const aclScalarList *scalarList, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| scalarList | 输入 | 输入的aclScalarList。 |
| size | 输出 | 返回aclScalarList的大小。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数scalarList或size为空指针。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclScalarList
std::vector<aclScalar *> tempscalar{alpha1, alpha2};
aclScalarList *scalarList = aclCreateScalarList(tempscalar.data(), tempscalar.size());
...
// 获取scalarList的大小
uint64_t size = 0;
auto ret = aclGetScalarListSize(scalarList, &size); // 这里获取到的scalarList的size为2
...
// 销毁aclScalarList
ret = aclDestroyScalarList(scalarList);
```
