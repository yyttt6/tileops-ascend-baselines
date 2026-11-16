# aclDestroyAclOpExecutor

## 功能说明

aclOpExecutor是框架定义的算子执行器，用来执行算子计算的容器，开发者无需关注其内部实现，直接使用即可。

- 对于非复用状态的aclOpExecutor，调用一阶段接口aclxxXxxGetWorkspaceSize时框架会自动创建aclOpExecutor，调用二阶段接口aclxxXxx时框架会自动释放aclOpExecutor，无需手动调用本接口释放。
- 对于复用状态的aclOpExecutor（调用[aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md)接口开启复用），算子执行器的管理由用户自行处理，因此aclOpExecutor的销毁需显式调用本接口手动销毁。

## 函数原型

```cpp
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor *executor)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| executor | 输入 | 待销毁的aclOpExecutor。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回561103：executor是空指针。

## 约束说明

本接口需与[aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md)接口配套使用，分别完成aclOpExecutor的复用与销毁。

## 调用示例

接口调用请参考[aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md)的调用示例。
