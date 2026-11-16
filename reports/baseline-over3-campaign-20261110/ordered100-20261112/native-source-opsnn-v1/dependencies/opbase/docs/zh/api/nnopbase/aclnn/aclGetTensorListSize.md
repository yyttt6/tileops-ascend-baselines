# aclGetTensorListSize

## 功能说明

获取aclTensorList的大小，aclTensorList通过[aclCreateTensorList](aclCreateTensorList.md)接口创建。

## 函数原型

```cpp
aclnnStatus aclGetTensorListSize(const aclTensorList *tensorList, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensorList | 输入 | 输入的aclTensorList。 |
| size | 输出 | 返回aclTensorList的大小。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数tensorList或size为空指针。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclTensorList
std::vector<aclTensor *> tmp{input1, input2};
aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
...
// 获取tensorList的大小
uint64_t size = 0;
auto ret = aclGetTensorListSize(tensorList, &size); // 这里获取到的tensorList的size为2
...
// 销毁aclTensorList
ret = aclDestroyTensorList(tensorList);
```
