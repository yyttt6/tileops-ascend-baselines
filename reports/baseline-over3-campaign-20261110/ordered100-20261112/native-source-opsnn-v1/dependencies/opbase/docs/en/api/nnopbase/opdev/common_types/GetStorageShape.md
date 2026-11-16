# GetStorageShape

## Function

Obtains `StorageShape` of an `aclTensor`.

StorageShape indicates the actual runtime shape of an `aclTensor` in memory. It reflects how `OriginShape` is rearranged under the selected `StorageFormat`.

## Prototype

```cpp
gert::Shape GetStorageShape()
```

## Parameters

None

## Returns

`gert::Shape`. It records a set of shape information, for example, a three-dimensional shape [10, 20, 30].

> **Note**: For details about `gert::Shape`, see "gert Namespace > Shape" in [Basic Data Structure and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

## Restrictions

None

## Examples

```cpp
void Func(const aclTensor *input) {
    auto shape = input->GetStorageShape();
}
```
