# OP\_LOGE\_WITH\_INVALID\_INPUT

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_VALUE](OP_LOGE_FOR_INVALID_VALUE.md) 或 [OP_CHECK_NULL_WITH_CONTEXT](OP_CHECK_NULL_WITH_CONTEXT.md) 替代。

## 功能说明

记录并上报算子输入参数为空校验错误。当算子的必需参数为空时，输出ERROR级别日志并上报EZ0004错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_INPUT(entityName, paramName)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter input of MyOp is required, but it is empty.
if (xDesc == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT("MyOp", "input");
    return false;
}
```
