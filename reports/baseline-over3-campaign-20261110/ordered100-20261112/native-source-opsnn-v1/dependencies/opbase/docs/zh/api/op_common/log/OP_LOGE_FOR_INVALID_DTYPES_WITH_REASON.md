# OP\_LOGE\_FOR\_INVALID\_DTYPES\_WITH\_REASON

## 功能说明

记录并上报多个参数数据类型校验错误（附带原因说明）。当算子的多个参数数据类型不正确时，输出ERROR级别日志（含具体原因）并上报EZ0021错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(entityName, paramNames, incorrectDtypes, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectDtypes | 输入 | 实际数据类型列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters cos and dy of RotaryPositionEmbeddingGrad have incorrect dtypes
//          FLOAT16 and FLOAT32. Reason: The dtypes of input cos and input dy should be the same.
if (cosDtype != dyDtype) {
    std::string dtypeMsg = Ops::Base::ToString(cosDtype) + " and " + Ops::Base::ToString(dyDtype);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("RotaryPositionEmbeddingGrad", "cos and dy",
        dtypeMsg.c_str(), "The dtypes of input cos and input dy should be the same.");
    return ge::GRAPH_FAILED;
}
```
