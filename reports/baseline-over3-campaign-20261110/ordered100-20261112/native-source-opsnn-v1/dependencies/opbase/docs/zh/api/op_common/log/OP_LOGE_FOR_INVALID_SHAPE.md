# OP\_LOGE\_FOR\_INVALID\_SHAPE

## 功能说明

记录并上报参数形状校验错误。当算子的指定参数形状与预期不符时，输出ERROR级别日志并上报EZ0008错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPE(entityName, paramName, incorrectShape, correctShape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectShape | 输入 | 实际形状，格式如"[1,128,128]"，支持const char*或std::string类型。 |
| correctShape | 输入 | 预期形状，格式如"[1,256,256]"，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter y of Ger has incorrect shape [128]. It should be [32,64].
if (yShape.GetDimNum() != 2) {
    std::string incorrectShape = Ops::Base::ToString(yShape);
    OP_LOGE_FOR_INVALID_SHAPE("Ger", "y", incorrectShape.c_str(), "[32,64]");
    return ge::GRAPH_FAILED;
}
```
