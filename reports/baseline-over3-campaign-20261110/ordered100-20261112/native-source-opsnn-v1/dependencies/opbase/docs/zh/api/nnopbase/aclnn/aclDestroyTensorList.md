# aclDestroyTensorList

## 功能说明

销毁通过[aclCreateTensorList](aclCreateTensorList.md)接口创建的aclTensorList。

## 函数原型

```cpp
aclnnStatus aclDestroyTensorList(const aclTensorList *array)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| array | 输入 | 需要销毁的aclTensorList。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

## 约束说明

对于aclTensorList内的aclTensor不需要重复释放。

## 调用示例

接口调用请参考[aclCreateTensorList](aclCreateTensorList.md)的调用示例。
