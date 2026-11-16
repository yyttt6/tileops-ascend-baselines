# aclGetFloatArraySize

## 功能说明

获取aclFloatArray的大小，aclFloatArray通过[aclCreateFloatArray](aclCreateFloatArray.md)接口创建。

## 函数原型

```cpp
aclnnStatus aclGetFloatArraySize(const aclFloatArray *array, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| array | 输入 | 输入的aclFloatArray。 |
| size | 输出 | 返回aclFloatArray的大小。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数array或size为空指针。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclFloatArray
std::vector<float> scalesData = {1.0, 1.0, 2.0, 2.0};
aclFloatArray *scales = aclCreateFloatArray(scalesData.data(), scalesData.size());
...
// 使用aclGetFloatArraySize接口获取scales的大小
uint64_t size = 0;
auto ret = aclGetFloatArraySize(scales, &size); // 获取到的scales的size为4
...
// 销毁aclFloatArray
ret = aclDestroyFloatArray(scales);
```
