# OP\_LOGE\_WITH\_INVALID\_INPUT\_SHAPESIZE

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_SHAPESIZE](OP_LOGE_FOR_INVALID_SHAPESIZE.md) 替代。

## 功能说明

记录并上报算子输入形状大小校验错误。当算子的指定输入形状大小（总元素数）与预期不符时，输出ERROR级别日志并上报EZ0005错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(entityName, index, incorrectSize, correctSize)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| index | 输入 | 输入张量索引（从0开始），int类型。 |
| incorrectSize | 输入 | 实际形状大小，支持const char*或std::string类型。 |
| correctSize | 输入 | 预期形状大小，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: The 0th input of MyOp has incorrect shape size 1024. It should be 2048.
if (xShape.GetShapeSize() != 2048) {
    OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE("MyOp", 0,
        std::to_string(xShape.GetShapeSize()), "2048");
    return false;
}
```
