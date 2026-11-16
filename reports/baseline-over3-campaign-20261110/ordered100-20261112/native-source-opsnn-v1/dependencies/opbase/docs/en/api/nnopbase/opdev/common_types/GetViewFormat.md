# GetViewFormat

## Function

Obtains the ViewFormat of an aclTensor. ViewFormat is the logical format of the aclTensor.

## Prototype

```cpp
op::Format GetViewFormat()
```

## Parameters

None

## Returns

`op::Format` (also known as `ge::Format`). It is an enumeration that defines different formats, such as NCHW and ND.

> **Note**: For details about `ge::Format`, see "ge Namespace > Format" in [Basic Data Structures and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

## Restrictions

None

## Examples

```cpp
void Func(const aclTensor *input) {
    auto format = input->GetViewFormat();
}
```
