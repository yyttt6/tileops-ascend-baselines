# OP\_LOGE\_FOR\_INVALID\_SHAPESIZE

## 功能说明

记录并上报参数形状大小校验错误。当算子的指定参数形状大小（总元素数）与预期不符时，输出ERROR级别日志并上报EZ0014错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPESIZE(entityName, paramName, incorrectSize, correctSize)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectSize | 输入 | 实际形状大小（总元素数），支持const char*或std::string类型。 |
| correctSize | 输入 | 预期形状大小（总元素数），支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter group_index of SwigluMxQuantWithDualAxis has incorrect shape size 0.
//          It should be > 0.
if (groupIndexShape.GetShapeSize() == 0) {
    OP_LOGE_FOR_INVALID_SHAPESIZE("SwigluMxQuantWithDualAxis", "group_index",
        std::to_string(groupIndexShape.GetShapeSize()), "> 0");
    return ge::GRAPH_FAILED;
}
```
