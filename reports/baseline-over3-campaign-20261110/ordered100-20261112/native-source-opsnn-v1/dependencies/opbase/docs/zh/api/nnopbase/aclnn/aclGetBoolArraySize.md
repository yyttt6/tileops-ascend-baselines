# aclGetBoolArraySize

## 功能说明

获取aclBoolArray的大小，aclBoolArray通过[aclCreateBoolArray](aclCreateBoolArray.md)接口创建。

## 函数原型

```cpp
aclnnStatus aclGetBoolArraySize(const aclBoolArray *array, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| array | 输入 | 输入的aclBoolArray。 |
| size | 输出 | 返回aclBoolArray的大小。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数array或size为空指针。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclBoolArray
bool maskData[] = {true, false};
aclBoolArray *mask = aclCreateBoolArray(maskData, sizeof(maskData) / sizeof(maskData[0]));
...
// 使用aclGetBoolArraySize接口获取mask的大小
uint64_t size = 0;
auto ret = aclGetBoolArraySize(mask, &size);     // 获取到的mask的size为2
...
// 销毁aclBoolArray
ret = aclDestroyBoolArray(mask);
```
