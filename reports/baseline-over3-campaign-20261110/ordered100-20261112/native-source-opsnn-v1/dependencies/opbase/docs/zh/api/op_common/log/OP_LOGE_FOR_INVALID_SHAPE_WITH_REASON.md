# OP\_LOGE\_FOR\_INVALID\_SHAPE\_WITH\_REASON

## 功能说明

记录并上报参数形状校验错误（附带原因说明）。当算子的指定参数形状不正确时，输出ERROR级别日志（含具体原因）并上报EZ0009错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entityName, paramName, incorrectShape, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectShape | 输入 | 实际形状，格式如"[1,128]"，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter indices 0th tensor of DynamicStitch has incorrect shape [2,-1,128].
//          Reason: The input indices's tensor has negative dimension.
if (hasNegativeDim) {
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("DynamicStitch",
        "indices 0th tensor", "[2,-1,128]",
        "The input indices's tensor has negative dimension.");
    return ge::GRAPH_FAILED;
}
```
