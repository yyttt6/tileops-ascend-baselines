# GetOriginalShape

## Function

Obtains `OriginShape` of an `aclTensor`.

`OriginShape` is the original shape of an `aclTensor` before it passes through a `transdata` node (if any), that is, the mathematical description of the tensor's shape.

## Prototype

```cpp
gert::Shape GetOriginalShape()
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
    auto shape = input->GetOriginalShape();
}
```
