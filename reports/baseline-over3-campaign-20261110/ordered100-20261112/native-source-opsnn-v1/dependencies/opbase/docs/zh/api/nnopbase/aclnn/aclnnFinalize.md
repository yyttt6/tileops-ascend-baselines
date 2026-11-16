# aclnnFinalize

## 功能说明

单算子API执行框架中aclnnXxx接口的去初始化函数，在进程退出前必须先释放进程内aclnn相关资源，否则会导致系统内部出错，影响业务正常运行。

> **说明**： 
>调用aclnnFinalize或aclFinalize接口，均能实现资源去初始化，二者区别在于aclnnFinalize仅完成aclnn相关资源释放，而aclFinalize完成acl接口中各种资源（包含aclnn）的释放。因此aclnnFinalize相对于aclFinalize更轻量一些。若两个接口都调用，也不返回失败。

## 函数原型

```cpp
aclnnStatus aclnnFinalize()
```

## 参数说明

无

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

## 约束说明

- 本接口需与[aclnnInit](aclnnInit.md)初始化接口配套使用，分别完成aclnn资源初始化与去初始化。
- 一个进程内只允许调用一次aclnnFinalize接口，不支持重复调用。

## 调用示例

接口调用请参考[aclnnInit](aclnnInit.md)的调用示例。
