# aclDestroyBoolArray

## 功能说明

销毁通过[aclCreateBoolArray](aclCreateBoolArray.md)接口创建的aclBoolArray。

## 函数原型

```cpp
aclnnStatus aclDestroyBoolArray(const aclBoolArray *array)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| array | 输入 | 需要销毁的aclBoolArray 。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

## 约束说明

无

## 调用示例

接口调用请参考[aclCreateBoolArray](aclCreateBoolArray.md)的调用示例。
