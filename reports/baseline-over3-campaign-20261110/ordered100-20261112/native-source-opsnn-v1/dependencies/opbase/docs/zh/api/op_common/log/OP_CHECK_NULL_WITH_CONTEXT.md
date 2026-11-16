# OP\_CHECK\_NULL\_WITH\_CONTEXT

## 功能说明

根据传入的context上下文，校验传入的指针是否为nullptr。

## 函数原型

```cpp
OP_CHECK_NULL_WITH_CONTEXT(context, ptr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| context | 输入 | 传入的上下文信息，类型为InferShapeContext/TilingParseContext/TilingContext。 |
| ptr | 输入 | 待判定的指针。 |

## 返回值说明

当输入ptr为nullptr时，返回ge::GRAPH\_FAILED。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
auto inShape = context->GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
auto axesTensor = context->GetInputTensor(1);
OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
auto outShape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
```
