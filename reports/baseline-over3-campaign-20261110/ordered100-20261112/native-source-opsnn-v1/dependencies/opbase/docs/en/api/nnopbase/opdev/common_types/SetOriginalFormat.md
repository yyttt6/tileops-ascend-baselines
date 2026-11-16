# SetOriginalFormat

## Function

Sets the OriginFormat of an aclTensor.

`OriginFormat` is the original format of an `aclTensor` before it passes through a `transdata` node (if any).

## Prototype

```cpp
void SetOriginalFormat(op::Format format)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| format | Input| The data type is `op::Format` (`ge::Format`). It is an enumeration that defines different formats, such as NCHW and ND.|

## Returns

None

## Restrictions

None

## Examples

```cpp
// Set the OriginFormat of the input to ND.
void Func(const aclTensor *input) {
    input->SetOriginalFormat(ge::FORMAT_ND);
}
```
