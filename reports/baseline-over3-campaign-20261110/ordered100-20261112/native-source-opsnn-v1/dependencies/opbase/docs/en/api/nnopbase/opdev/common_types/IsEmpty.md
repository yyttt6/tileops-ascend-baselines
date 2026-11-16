# IsEmpty

## Function

Checks if an `aclTensor` is an empty tensor.

## Prototype

```cpp
bool IsEmpty()
```

## Parameters

None

## Returns

If an `aclTensor` is an empty tensor, **true** is returned. Otherwise, **false** is returned.

## Restrictions

None

## Examples

```cpp
// If the input is an empty tensor, the function returns. Otherwise, its data type is returned.
void Func(const aclTensor *input) {
    if (input->IsEmpty()) {
        return;
    }
    op::DataType dataType = input->GetDataType();
}
```
