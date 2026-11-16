# OP\_LOGE\_FOR\_INVALID\_SHAPEDIMS\_WITH\_REASON

## 功能说明

记录并上报多个参数形状维度校验错误（附带原因说明）。当算子的多个参数形状维度不正确时，输出ERROR级别日志（含具体原因）并上报EZ0013错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(entityName, paramNames, incorrectDims, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectDims | 输入 | 实际维度列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters dy, cos and sin of RotaryPositionEmbeddingGrad have incorrect shape dims
//          3, 4 and 4. Reason: The numbers of dimensions of input dy, cos and sin should all be 3D or 4D.
if (dyDimNum != expectedDim || cosDimNum != expectedDim || sinDimNum != expectedDim) {
    std::string dimMsg = std::to_string(dyDimNum) + ", " + std::to_string(cosDimNum) +
        " and " + std::to_string(sinDimNum);
    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("RotaryPositionEmbeddingGrad",
        "dy, cos and sin", dimMsg.c_str(),
        "The numbers of dimensions of input dy, cos and sin should all be 3D or 4D.");
    return ge::GRAPH_FAILED;
}
```
