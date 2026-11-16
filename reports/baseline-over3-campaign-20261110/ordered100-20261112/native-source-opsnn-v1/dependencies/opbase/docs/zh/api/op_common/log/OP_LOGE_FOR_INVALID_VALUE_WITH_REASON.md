# OP\_LOGE\_FOR\_INVALID\_VALUE\_WITH\_REASON

## 功能说明

记录并上报参数值校验错误（附带原因说明）。当算子的指定参数值不正确时，输出ERROR级别日志（含具体原因）并上报EZ0026错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entityName, paramName, incorrectValue, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectValue | 输入 | 实际参数值，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter output_size of UpsampleNearest3D has incorrect value
//          (2147483648, 512, 512). Reason: Each value of output_size must be less than or equal to
//          INT32_MAX.
if (outputSizeVal > INT32_MAX) {
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("UpsampleNearest3D", "output_size",
        outputSizeStr.c_str(),
        "Each value of output_size must be less than or equal to INT32_MAX.");
    return ge::GRAPH_FAILED;
}
```
