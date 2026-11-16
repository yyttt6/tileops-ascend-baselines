# OP\_OUTPUT

## Function

Encapsulates an operator's outputs, including both `aclTensor` and `aclTensorList`.

## Prototype

```cpp
OP_OUTPUT(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| Output `aclTensor` and `aclTensorList`.|

## Restrictions

If an operator's output is neither `aclTensor` nor `aclTensorList`, call `aclOpExecutor::ConvertToTensor` to convert it to `aclTensor`.

## Examples

```cpp
// Encapsulate an operator's output: addOut.
OP_OUTPUT(addOut);
```
