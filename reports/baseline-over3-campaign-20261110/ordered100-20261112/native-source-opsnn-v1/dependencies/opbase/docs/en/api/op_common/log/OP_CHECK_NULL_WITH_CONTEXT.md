# OP\_CHECK\_NULL\_WITH\_CONTEXT

## Function

Checks if an input pointer is `nullptr` based on the input context.

## Prototype

```cpp
OP_CHECK_NULL_WITH_CONTEXT(context, ptr)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| context | Input| Input context, which is of type `InferShapeContext`, `TilingParseContext`, or `TilingContext`.|
| ptr | Input| Pointer to be checked.|

## Returns

If the input `ptr` is `nullptr`, ge::GRAPH\_FAILED is returned.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
auto inShape = context->GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
auto axesTensor = context->GetInputTensor(1);
OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
auto outShape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
```
