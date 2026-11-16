# OP\_LOGE\_FOR\_INVALID\_SHAPESIZES\_WITH\_REASON

## 功能说明

记录并上报多个参数形状大小校验错误（附带原因说明）。当算子的多个参数形状大小不正确时，输出ERROR级别日志（含具体原因）并上报EZ0016错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(entityName, paramNames, incorrectSizes, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectSizes | 输入 | 实际形状大小列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters query, key, cos and sin of ApplyRotaryPosEmb have incorrect shape sizes
//          0, 0, 0 and 0. Reason: All inputs must be non-empty tensors.
if (qSize == 0 || kSize == 0 || cSize == 0 || sSize == 0) {
    std::string sizeMsg = std::to_string(qSize) + ", " + std::to_string(kSize) + ", " +
        std::to_string(cSize) + " and " + std::to_string(sSize);
    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON("ApplyRotaryPosEmb",
        "query, key, cos and sin", sizeMsg.c_str(),
        "All inputs must be non-empty tensors.");
    return ge::GRAPH_FAILED;
}
```
