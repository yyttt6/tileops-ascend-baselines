# ToShapeVector

## Function

Extracts the dimension information from the input shape object to a ShapeVector container.

## Prototype

```cpp
FVector<int64_t, 25> ToShapeVector(const op::Shape &shape)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| shape | Input| Shape information, for example, a three-dimensional shape: [10, 20, 30].|

## Returns

An FVector object that stores the size of each dimension

## Restrictions

None

## Examples

```cpp
// Generate a shape object whose shape is [1, 2, 3, 4, 5] and convert it to an FVector.
void Func() {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    auto shapeVec = ToShapeVector(newShape);
}
```
