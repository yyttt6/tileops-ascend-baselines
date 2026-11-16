# aclGetIntArraySize

## 功能说明

获取aclIntArray的大小，aclIntArray通过[aclCreateIntArray](aclCreateIntArray.md)接口创建。

## 函数原型

```cpp
aclnnStatus aclGetIntArraySize(const aclIntArray *array, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| array | 输入 | 输入的aclIntArray。 |
| size | 输出 | 返回aclIntArray的大小。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数array或size为空指针。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclIntArray
std::vector<int64_t> valueData = {1, 1, 2, 3};
aclIntArray *valueArray = aclCreateIntArray(valueData.data(), valueData.size());
...
// 使用aclGetIntArraySize接口获取valueArray的大小
uint64_t size = 0;
auto ret = aclGetIntArraySize(valueArray, &size); // 获取到的valueArray的size为4
...
// 销毁aclIntArray
ret = aclDestroyIntArray(valueArray);
```
