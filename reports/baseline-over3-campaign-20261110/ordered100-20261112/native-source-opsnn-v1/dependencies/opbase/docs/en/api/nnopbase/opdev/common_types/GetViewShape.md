# GetViewShape

## Function

Obtains `ViewShape` of an `aclTensor`. ViewShape indicates the logical shape of an `aclTensor`.

Assume that `StorageShape` and `ViewShape` of an `aclTensor` are as follows:

- If `StorageShape` is \[10, 20\], the `aclTensor` is arranged in memory with dimensions \[10, 20\].
- If `ViewShape` is \[2, 5, 20\], during operator execution, the `aclTensor` can be treated as data arranged in the shape \[2, 5, 20\].

## Prototype

```cpp
gert::Shape GetViewShape()
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
    auto shape = input->GetViewShape();
}
```
