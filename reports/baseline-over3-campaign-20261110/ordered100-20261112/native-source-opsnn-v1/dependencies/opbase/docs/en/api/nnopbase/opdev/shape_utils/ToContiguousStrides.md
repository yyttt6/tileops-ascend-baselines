# ToContiguousStrides

## Function

Derives stride information from the given shape by following the contiguous memory layout rule. For example, if the shape is \[10, 20, 30\], the derived stride is \[600, 30, 1\].

## Prototype

```cpp
void ToContiguousStrides(const op::Shape &shape, op::Strides &strides)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| A set of shape information, for example, a three-dimensional shape: [10, 20, 30].|
| strides | Output| A set of stride information, for example, the stride of a three-dimensional shape: [600, 30, 1].|

## Returns

None

## Restrictions

None

## Examples

```cpp
// Generate a shape object with shape information is [1, 2, 3, 4, 5] and generate its stride data.
void Func() {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    FVector<int64_t, 25> strides;
    ToContiguousStrides(newShape, strides);
}
```
