# SetOriginalShape

## Function

Sets the OriginShape attribute of an aclTensor.

`OriginShape` is the original shape of an `aclTensor` before it passes through a `transdata` node (if any), that is, the mathematical description of the tensor's shape.

## Prototype

```cpp
void SetOriginalShape(const op::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| The data type is `op::Shape` (that is, `gert::Shape`), which records a group of shape information, for example, a three-dimensional shape: [10, 20, 30].|

## Returns

None

## Restrictions

None

## Examples

```cpp
// Set OriginShape of the input to [1, 2, 3, 4, 5].
void Func(aclTensor *input) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    input->SetOriginalShape(newShape);
}
```
