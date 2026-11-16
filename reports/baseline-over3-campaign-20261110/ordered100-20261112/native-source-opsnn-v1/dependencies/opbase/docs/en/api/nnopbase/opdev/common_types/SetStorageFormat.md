# SetStorageFormat

## Function

Sets `StorageFormat` of an `aclTensor`.

`StorageFormat` specifies how an `aclTensor` is arranged in memory, such as NCHW or ND.

## Prototype

```cpp
void SetStorageFormat(op::Format format)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| format | Input| The data type is `op::Format` (that is, `ge::Format`). It is an enumeration that defines different formats, such as NCHW and ND.|

## Returns

None

## Restrictions

None

## Examples

```cpp
// Set the storage format of the input to ND.
void Func(const aclTensor *input) {
    input->SetStorageFormat(ge::FORMAT_ND);
}
```
