# OP\_LOGE\_FOR\_INVALID\_VALUES\_WITH\_REASON

## 功能说明

记录并上报多个参数值校验错误（附带原因说明）。当算子的多个参数值不正确时，输出ERROR级别日志（含具体原因）并上报EZ0027错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entityName, paramNames, incorrectValues, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectValues | 输入 | 实际参数值列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters align_corners and half_pixel_centers of ResizeNearestNeighborV2Grad have
//          incorrect values true and true. Reason: The values of attributes align_corners and
//          half_pixel_centers cannot be true at the same time.
if (alignCorners_ && halfPixelCenters_) {
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("ResizeNearestNeighborV2Grad",
        "align_corners and half_pixel_centers", "true and true",
        "The values of attributes align_corners and half_pixel_centers cannot be true at the same time.");
    return ge::GRAPH_FAILED;
}
```
