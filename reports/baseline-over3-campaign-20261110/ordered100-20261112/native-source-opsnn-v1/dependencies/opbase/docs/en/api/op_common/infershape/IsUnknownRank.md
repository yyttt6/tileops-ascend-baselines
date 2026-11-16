# IsUnknownRank

## Function

Checks if an input shape is of unknown rank in graph mode.

## Prototype

```cpp
bool IsUnknownRank(const gert::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| Input shape to be checked.|

## Returns

Return type: `bool`

- **true**: The input shape is of unknown rank.
- **false**: The input shape is not of unknown rank.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
auto in_shape = context->GetInputShape(0); // 0 indicates the first input parameter.
OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
auto out_shape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
// Check if the shape of the input tensor is of unknown rank. If yes, the shape of the output tensor is set to unknown rank.
if (Ops::Base::IsUnknownRank(*in_shape)) {
    Ops::Base::SetUnknownRank(*out_shape);
}
```
