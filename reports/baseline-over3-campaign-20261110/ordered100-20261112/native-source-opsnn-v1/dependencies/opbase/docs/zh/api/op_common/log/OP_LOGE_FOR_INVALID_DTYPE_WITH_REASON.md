# OP\_LOGE\_FOR\_INVALID\_DTYPE\_WITH\_REASON

## 功能说明

记录并上报参数数据类型校验错误（附带原因说明）。当算子的指定参数数据类型不正确时，输出ERROR级别日志（含具体原因）并上报EZ0020错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(entityName, paramName, incorrectDtype, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectDtype | 输入 | 实际数据类型（如"INT8"），支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter k_cache of KvRmsNormRopeCache has incorrect dtype FLOAT16.
//          Reason: The dtype of input k_cache should be INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2,
//          or the same as the dtype of input kv.
if (!IsSupportedDtype(kCacheDtype, kvDtype)) {
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("KvRmsNormRopeCache", "k_cache",
        Ops::Base::ToString(kCacheDtype).c_str(),
        "The dtype of input k_cache should be INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2, "
        "or the same as the dtype of input kv.");
    return ge::GRAPH_FAILED;
}
```
