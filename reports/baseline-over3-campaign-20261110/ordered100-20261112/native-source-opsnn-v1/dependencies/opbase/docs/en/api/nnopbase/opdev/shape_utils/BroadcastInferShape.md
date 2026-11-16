# BroadcastInferShape

## Function

Infers the broadcast shape for two shapes that meet the broadcast relationship. For details about the broadcast rules, see [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html). For example, the broadcast shape of \[1, 10\] and \[2, 1\] is \[2, 10\].

## Prototype

```cpp
bool BroadcastInferShape(const op::Shape &self, const op::Shape &other, op::Shape &broadcastShape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| self | Input| First shape.|
| other | Input| Second shape.|
| broadcastShape | Output| Broadcast shape for `self` and `other`.|

## Returns

`true` is returned if `self` and `other` satisfy broadcasting rules. Otherwise, `false` is returned.

## Restrictions

None

## Examples

```cpp
// Generate two shape objects with shapes [2, 1] and [2, 10] to obtain the broadcast shape.
void Func() {
    gert::Shape shapeA;
    shapeA.AppendDim(1);
    shapeA.AppendDim(2);
    gert::Shape shapeB;
    shapeB.AppendDim(10);
    shapeB.AppendDim(2);
    gert::Shape shapeBrc;
    bool isBrc = BroadcastInferShape(shapeA, shapeB, shapeBrc);
}
```
