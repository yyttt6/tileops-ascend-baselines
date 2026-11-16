# IsUnknownShape

## Function

Checks if each dimension length of an input shape is unknown in graph mode.

## Prototype

```cpp
bool IsUnknownShape(const gert::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| Input shape to be checked.|

## Returns

Return type: `bool`

- **true**: Each dimension length of the input shape is unknown.
- **false**: At least one dimension length of the input shape is known.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
const gert::Shape* inputShape = context->GetInputShape(DIAGFLAT_IN_X_IDX);
OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
if (Ops::Base::IsUnknownShape(*inputShape)) {
    output_y_shape->SetDimNum(input_dim_num);
}
```
