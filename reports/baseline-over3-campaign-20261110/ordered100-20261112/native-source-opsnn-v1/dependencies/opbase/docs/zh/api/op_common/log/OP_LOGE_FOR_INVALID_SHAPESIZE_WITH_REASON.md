# OP\_LOGE\_FOR\_INVALID\_SHAPESIZE\_WITH\_REASON

## 功能说明

记录并上报参数形状大小校验错误（附带原因说明）。当算子的指定参数形状大小不正确时，输出ERROR级别日志（含具体原因）并上报EZ0015错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(entityName, paramName, incorrectSize, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectSize | 输入 | 实际形状大小（总元素数），支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter y of ResizeLinear has incorrect shape size [144,144,1].
//          Reason: The linear-dimension of output y must be equal to value (256) of input parameter size.
if (yShape.GetShapeSize() != expectedSize) {
    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("ResizeLinear", "y",
        Ops::Base::ToString(yShape),
        "The linear-dimension of output y must be equal to value (256) of input parameter size.");
    return ge::GRAPH_FAILED;
}
```
