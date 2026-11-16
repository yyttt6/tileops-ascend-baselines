# TypeSize

## Function

Calculates how many bytes a single element of the input data type occupies. For a data type smaller than one byte, `1000` will be added to the base bit size. For example, `Int4` will return `1004`, indicating that one element occupies 4 bits.

## Prototype

```cpp
size_t TypeSize(DataType dataType)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| dataType | Input| Input data type. The size of a single element of this data type is returned.|

## Returns

Size of a single element of the input data type.

## Restrictions

None

## Examples

```cpp
// Obtain the size of a single element of the specified data type.
void Func(const DataType dtype) {
    size_t size = TypeSize(dtype);
}
```
