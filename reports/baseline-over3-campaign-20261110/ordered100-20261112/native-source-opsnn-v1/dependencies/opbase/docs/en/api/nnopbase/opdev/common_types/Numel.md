# Numel

## Function

Obtains the total number of elements in an `aclTensor`.

## Prototype

```cpp
int64_t Numel()
```

## Parameters

None

## Returns

Total number of elements in an `aclTensor`.

## Restrictions

None

## Examples

```cpp
// Obtain the total number of input elements.
void Func(const aclTensor *input) {
    int64_t num = input->Numel();
}
```
