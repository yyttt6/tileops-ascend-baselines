# OP\_CHECK\_IF

## Function

Outputs a log and executes the `return` expression if the condition is true.

## Prototype

```cpp
OP_CHECK_IF(condition, log, return_expr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| condition | Input| Condition check.|
| log | Input| Log to be output. The log is printed using `OP_LOGE`.|
| return_expr | Input| `return` expression.|

## Returns

None

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
auto axesTensor = context->GetInputTensor(1);
OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
auto axesSize = static_cast<int32_t>(axesTensor->GetShapeSize());

OP_CHECK_IF(axesSize < 0, OP_LOGE(context->GetNodeName(), "axes num cannot be less than 0!"), return ge::GRAPH_FAILED);
```
