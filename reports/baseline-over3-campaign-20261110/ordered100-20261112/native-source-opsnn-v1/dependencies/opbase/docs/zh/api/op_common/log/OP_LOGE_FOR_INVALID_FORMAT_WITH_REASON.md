# OP\_LOGE\_FOR\_INVALID\_FORMAT\_WITH\_REASON

## 功能说明

记录并上报参数格式校验错误（带原因）。当算子的指定参数格式无效时，输出ERROR级别日志并上报EZ0035错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(entityName, paramName, incorrectFormat, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectFormat | 输入 | 实际格式，如"NHWC"，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (!IsSupportedFormat(format)) {
    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("MyOp", "x", "NDHWC", "only support NCHW and NHWC");
    return ge::GRAPH_FAILED;
}
```
