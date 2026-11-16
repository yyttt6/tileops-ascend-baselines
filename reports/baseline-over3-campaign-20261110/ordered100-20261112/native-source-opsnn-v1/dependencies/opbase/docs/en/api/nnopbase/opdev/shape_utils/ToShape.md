# ToShape

## Function

Constructs an `op::Shape` (that is, `gert::Shape`) object from the specified dimension data in the memory.

## Prototype

- Construct `op::Shape` using the given `dims` dimension array.

    ```cpp
    void ToShape(const int64_t *dims, uint64_t dimNum, op::Shape &shape)
    ```

- Construct `op::Shape` using the given `op::ShapeVector` dimension array.

    ```cpp
    void ToShape(const op::ShapeVector &shapeVector, op::Shape &shape)
    ```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dims | Input| Source data pointer, pointing to the memory that stores the shape size of each dimension.|
| dimNum | Input| Number of dimensions contained in the source data that `dims` points to.|
| shapeVector | Input| Source data. It lists the shape size of each dimension.|
| shape | Output| Shape object constructed from the source data.|

## Returns

None

## Restrictions

None

## Examples

```cpp
// Generate a shape object with shape [1, 2, 3, 4, 5].
void Func() {
    int64_t myArray[5] = {1, 2, 3, 4, 5};
    Shape shape;
    ToShape(myArray, 5, shape);
}

```
