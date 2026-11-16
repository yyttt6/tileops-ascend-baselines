# OP\_CHECK\_IF

## 功能说明

当condition条件成立时，输出日志，并执行return表达式。

## 函数原型

```cpp
OP_CHECK_IF(condition, log, return_expr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| condition | 输入 | 条件校验。 |
| log | 输入 | 输出日志，使用OP_LOGE。 |
| return_expr | 输入 | return表达式。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
auto axesTensor = context->GetInputTensor(1);
OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
auto axesSize = static_cast<int32_t>(axesTensor->GetShapeSize());

OP_CHECK_IF(axesSize < 0, OP_LOGE(context->GetNodeName(), "axes num cannot be less than 0!"), return ge::GRAPH_FAILED);
```
