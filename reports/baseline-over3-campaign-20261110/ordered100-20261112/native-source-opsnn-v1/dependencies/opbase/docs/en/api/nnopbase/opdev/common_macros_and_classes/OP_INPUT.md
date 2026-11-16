# OP\_INPUT

## Function

Encapsulates the inputs aclTensor and aclTensorList of the operator.

## Prototype

```cpp
OP_INPUT(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| Inputs aclTensor and aclTensorList.|

## Restrictions

If the operator contains non-aclTensor and non-aclTensorList input parameters, call `aclOpExecutor::ConvertToTensor` to convert the parameters to aclTensor.

## Examples

```cpp
// Encapsulate the two input parameters of the operator: self and other.
OP_INPUT(self, other);
```
