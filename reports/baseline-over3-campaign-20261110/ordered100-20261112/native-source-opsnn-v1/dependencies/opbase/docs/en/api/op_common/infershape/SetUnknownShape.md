# SetUnknownShape

## Function

Sets the input tensor shape to a known rank and marks each dimension length as unknown in graph mode.

## Prototype

```cpp
void SetUnknownShape(int64_t rank, gert::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| rank | Input| Shape rank.|
| shape | Output| The shape is set to a known rank, with each dimension length marked as unknown.|

## Returns

None

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
auto in_shape = context->GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
auto out_shape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

if (Ops::Base::IsUnknownShape(*in_shape)) {
    Ops::Base::SetUnknownShape(0, *out_shape);
}
```
