# InferShape4Broadcast

## Function

Infers an output tensor's shape. It is the `InferShape` for `Broadcast` operators in graph mode.

## Prototype

```cpp
ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| context | Input| `InferShape` context passed by GE.|

## Returns

The return type is `ge::graphStatus`:

- ge::GRAPH\_SUCCESS: `InferShape` succeeded.
- ge::GRAPH\_FAIL: `InferShape` failed.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
// Apply inferShape to the IsFinite operator. The procedure follows the same process as Broadcast operators, allowing direct reuse.
ge::graphStatus InferShape4IsFinite(gert::InferShapeContext* context)
{
    return Ops::Base::InferShape4Broadcast(context);
}
// Register the IsFinite operator and its shape inference function with GE.
IMPL_OP_INFERSHAPE(IsFinite).InferShape(InferShape4IsFinite);
}
```
