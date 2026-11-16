# OP\_LOGE\_FOR\_INVALID\_SHAPEDIM\_WITH\_REASON

## 功能说明

记录并上报参数形状维度校验错误（附带原因说明）。当算子的指定参数形状维度不正确时，输出ERROR级别日志（含具体原因）并上报EZ0012错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(entityName, paramName, incorrectDim, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectDim | 输入 | 实际维度（如"2D"），支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter query of ApplyRotaryPosEmb has incorrect shape dim 3D.
//          Reason: The shape dims of input query must be 4 when the attr layout is 1 (BSND).
if (queryDimNum != 4 && layout_ == 1) {
    std::string dimStr = std::to_string(queryDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("ApplyRotaryPosEmb", "query",
        dimStr.c_str(),
        "The shape dims of input query must be 4 when the attr layout is 1 (BSND).");
    return ge::GRAPH_FAILED;
}
```
