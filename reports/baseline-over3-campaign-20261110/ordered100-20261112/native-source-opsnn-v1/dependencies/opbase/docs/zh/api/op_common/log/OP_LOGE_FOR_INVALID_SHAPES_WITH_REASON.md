# OP\_LOGE\_FOR\_INVALID\_SHAPES\_WITH\_REASON

## 功能说明

记录并上报多个参数形状校验错误（附带原因说明）。当算子的多个参数形状不正确时，输出ERROR级别日志（含具体原因）并上报EZ0010错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entityName, paramNames, incorrectShapes, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectShapes | 输入 | 实际形状列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters indices 0th tensor and x 0th tensor of DynamicStitch have incorrect shapes
//          [2,3] and [4,5,6]. Reason: The shape of indices's tensor should match the shape formed by
//          the first 2 axes of x's corresponding tensor.
if (indicesShape != expectedShape) {
    std::string shapeMsg = Ops::Base::ToString(indicesShape) + " and " + Ops::Base::ToString(xShape);
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("DynamicStitch",
        "indices 0th tensor and x 0th tensor", shapeMsg.c_str(),
        "The shape of indices's tensor should match the shape formed by the "
        "first 2 axes of x's corresponding tensor.");
    return ge::GRAPH_FAILED;
}
```
