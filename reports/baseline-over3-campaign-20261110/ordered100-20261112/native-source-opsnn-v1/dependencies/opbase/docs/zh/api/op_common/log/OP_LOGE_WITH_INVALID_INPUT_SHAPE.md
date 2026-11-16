# OP\_LOGE\_WITH\_INVALID\_INPUT\_SHAPE

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_SHAPE](OP_LOGE_FOR_INVALID_SHAPE.md) 替代。

## 功能说明

记录并上报算子输入形状校验错误。当算子的指定输入张量形状与预期不符时，输出ERROR级别日志并上报EZ0001错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_INPUT_SHAPE(entityName, index, incorrectShape, correctShape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| index | 输入 | 输入张量索引（从0开始），int类型。 |
| incorrectShape | 输入 | 实际输入形状，格式如"[1,128,128]"，支持const char*或std::string类型。 |
| correctShape | 输入 | 预期输入形状，格式如"[1,256,256]"，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: The 0th input of MatMul has incorrect shape [128,128]. It should be [256,512].
const gert::Shape expectedShape({256, 512});
if (inputShape != expectedShape) {
    std::string incorrectShape = Ops::Base::ToString(inputShape);
    OP_LOGE_WITH_INVALID_INPUT_SHAPE("MatMul", 0, incorrectShape.c_str(), "[256,512]");
    return ge::GRAPH_FAILED;
}
```
