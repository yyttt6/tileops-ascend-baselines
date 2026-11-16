# OP\_LOGE\_FOR\_INVALID\_FORMATS\_WITH\_REASON

## 功能说明

记录并上报多个参数格式校验错误（附带原因说明）。当算子的多个参数格式不正确时，输出ERROR级别日志（含具体原因）并上报EZ0018错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(entityName, paramNames, incorrectFormats, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectFormats | 输入 | 实际格式列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters x and y of OperatorName have incorrect formats ND and NCHW.
//          Reason: The formats of all inputs must match.
if (xFormat != yFormat) {
    std::string formatMsg = Ops::Base::ToString(xFormat) + " and " + Ops::Base::ToString(yFormat);
    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("OperatorName", "x and y",
        formatMsg.c_str(), "The formats of all inputs must match.");
    return ge::GRAPH_FAILED;
}
```
