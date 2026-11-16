# CheckBroadcastShape

## Function

Checks if two tensor shapes satisfy broadcasting rules. For details about the rules, see the [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) official website. For example, \[2, 1\] and \[2, 10\] satisfy the rules, but \[2, 2\] and \[2, 10\] do not.

## Prototype

```cpp
bool CheckBroadcastShape(const op::Shape &self, const op::Shape &other)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| self | Input| First shape.|
| other | Input| Second shape.|

## Returns

**true** is returned if `self` and `other` satisfy broadcasting rules. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// Generate two shapes [2, 1] and [2, 10] and check if they satisfy broadcasting rules.
void Func() {
    gert::Shape shapeA;
    shapeA.AppendDim(1);
    shapeA.AppendDim(2);
    gert::Shape shapeB;
    shapeB.AppendDim(10);
    shapeB.AppendDim(2);
    bool isBrc = CheckBroadcastShape(shapeA, shapeB);
}
```
