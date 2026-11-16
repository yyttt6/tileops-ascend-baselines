# GetViewStrides

## Function

Obtains `ViewStrides` of an `aclTensor`.

## Prototype

```cpp
FVector<int64_t> GetViewStrides()
```

## Parameters

None

## Returns

An `FVector` object. It stores the stride of each dimension of the `aclTensor`.

## Restrictions

None

## Examples

```cpp
// Obtain the view stride of the input and print the stride of each dimension in sequence.
void Func(const aclTensor *input) {
    auto strides = input->GetViewStrides();
    for (int64_t stride : strides) {
        std::cout << stride << std::endl;
    }
}
```
