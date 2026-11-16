# IsContiguous

## Function

Checks if an input tensor is contiguous.

## Prototype

```cpp
bool IsContiguous(const aclTensor *tensor)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| Tensor to be checked.|

## Returns

If the tensor is contiguous, **true** is returned. Otherwise, **false** is returned.

## Restrictions

- The tensor must not be empty.
- In the case of a `null` pointer, **true** is returned and the following error message is displayed: check tensor != nullptr failed.

## Examples

```cpp
// If the tensor is not contiguous, the function returns.
void Func(aclTensor *tensor) {
    if (!IsContiguous(tensor)) {
        return;
    }
}
```
