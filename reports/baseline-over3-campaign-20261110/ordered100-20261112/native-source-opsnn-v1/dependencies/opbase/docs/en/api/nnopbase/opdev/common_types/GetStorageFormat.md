# GetStorageFormat

## Function

Obtains `StorageFormat` of an `aclTensor`.

`StorageFormat` specifies how an `aclTensor` is arranged in memory, such as NCHW or ND.

## Prototype

```cpp
op::Format GetStorageFormat()
```

## Parameters

None

## Returns

`op::Format` (also known as `ge::Format`). It is an enumeration that defines different formats, such as NCHW and ND.

> **Note**: **: For details about `ge::Format`, see "ge Namespace > Format" in [Basic Data Structure and API Reference](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi).

## Restrictions

None

## Examples

```cpp
void Func(const aclTensor *input) {
    auto format = input->GetStorageFormat();
}
```
