# GetOriginalFormat

## Function

Obtains `OriginFormat` of an `aclTensor`.

`OriginFormat` is the original format of an `aclTensor` before it passes through a `transdata` node (if any).

## Prototype

```cpp
op::Format GetOriginalFormat()
```

## Parameters

None

## Returns

`op::Format` (also known as `ge::Format`). It is an enumeration that defines different formats, such as NCHW and ND.

> **Note**: For details about `ge::Format`, see "ge Namespace > Format" in [Basic Data Structure and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

## Restrictions

None

## Examples

```cpp
void Func(const aclTensor *input) {
    auto format = input->GetOriginalFormat();
}
```
