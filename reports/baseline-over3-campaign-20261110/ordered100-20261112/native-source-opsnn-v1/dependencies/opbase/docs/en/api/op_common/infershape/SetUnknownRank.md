# SetUnknownRank

## Function

Sets the output tensor's shape to unknown rank if the input tensor has an unknown rank in graph mode.

## Prototype

```cpp
void SetUnknownRank(gert::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Output| Shape of the output tensor.|

## Returns

None

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
auto in_shape = context->GetInputShape(0); // 0 indicates the first input parameter.
OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
auto out_shape = context->GetOutputShape(0); // 0 indicates the first output parameter.
OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

// Check if the shape of the input tensor is of unknown rank. If yes, the shape of the output tensor is set to unknown rank.
if (Ops::Base::IsUnknownRank(*in_shape)) {
    Ops::Base::SetUnknownRank(*out_shape);
}
```
