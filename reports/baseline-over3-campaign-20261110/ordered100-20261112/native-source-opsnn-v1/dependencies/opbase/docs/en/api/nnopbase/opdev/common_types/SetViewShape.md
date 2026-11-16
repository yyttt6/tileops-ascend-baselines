# SetViewShape

## Function

Sets `ViewShape` of an `aclTensor`. `ViewShape` indicates the logical shape of an `aclTensor` and can be obtained by calling [GetViewShape](GetViewShape.md).

## Prototype

```cpp
void SetViewShape(const op::Shape &shape)
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
// Set ViewShape of the input to [1, 2, 3, 4, 5].
void Func(aclTensor *input) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    input->SetViewShape(newShape);
}
```
