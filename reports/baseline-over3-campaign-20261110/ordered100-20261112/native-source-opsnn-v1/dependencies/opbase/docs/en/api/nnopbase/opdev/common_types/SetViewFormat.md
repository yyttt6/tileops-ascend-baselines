# SetViewFormat

## Function

Sets `ViewFormat` of an `aclTensor`. `ViewFormat` indicates the logical format of an `aclTensor`.

## Prototype

```cpp
void SetViewFormat(op::Format format)
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
// Set ViewFormat of the input to ND.
void Func(const aclTensor *input) {
    input->SetViewFormat(ge::FORMAT_ND);
}
```
