# OP\_LOGE\_WITH\_INVALID\_ATTR

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_VALUE](OP_LOGE_FOR_INVALID_VALUE.md) 替代。

## 功能说明

记录并上报算子属性值校验错误。当算子的指定属性值与预期不符时，输出ERROR级别日志并上报EZ0002错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_ATTR(entityName, attrName, incorrectVal, correctVal)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| attrName | 输入 | 属性名称，支持const char*或std::string类型。 |
| incorrectVal | 输入 | 实际属性值，支持const char*或std::string类型。 |
| correctVal | 输入 | 预期属性值，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Attribute axis of MyOp has incorrect value 3. It should be 0 or 1.
if (axis_ != "0" && axis_ != "1") {
    OP_LOGE_WITH_INVALID_ATTR("MyOp", "axis", std::to_string(axis_), "0 or 1");
    return false;
}
```
