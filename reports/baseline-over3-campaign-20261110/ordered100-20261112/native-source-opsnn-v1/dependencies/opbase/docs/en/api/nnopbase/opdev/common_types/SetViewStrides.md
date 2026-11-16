# SetViewStrides

## Function

Sets `ViewStrides` of an `aclTensor`.

## Prototype

```cpp
SetViewStrides(const op::Strides &strides)
```

```cpp
SetViewStrides(op::Strides &&strides)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| strides | Input| Stride size of each `aclTensor` dimension. The type is `FVector<int64_t>`. The value cannot be a negative number.|

## Returns

void

## Restrictions

None

## Examples

```cpp
// Multiply the input's last dimension stride by 8.
void Func(const aclTensor *input) {
    auto strides = input->GetViewStrides();
    strides[strides.size() - 1] *= 8;
    input->SetViewStrides(strides);
}
```
