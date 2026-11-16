# OP\_LOGE\_WITH\_INVALID\_ATTR\_SIZE

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_LISTSIZE](OP_LOGE_FOR_INVALID_LISTSIZE.md) 替代。

## 功能说明

记录并上报算子属性大小校验错误。当算子的指定属性大小与预期不符时，输出ERROR级别日志并上报EZ0003错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_ATTR_SIZE(entityName, attrName, incorrectSize, correctSize)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| attrName | 输入 | 属性名称，支持const char*或std::string类型。 |
| incorrectSize | 输入 | 实际属性大小，支持const char*或std::string类型。 |
| correctSize | 输入 | 预期属性大小，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Attribute strides of MyOp has incorrect size 3. It should be 4.
if (strides.size() != 4) {
    OP_LOGE_WITH_INVALID_ATTR_SIZE("MyOp", "strides", std::to_string(strides.size()), "4");
    return false;
}
```
