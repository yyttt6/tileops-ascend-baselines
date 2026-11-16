# CheckOverflows

## Function

Checks whether overflow occurs in the scalar stored in aclScalar.

## Prototype

```cpp
template<typename to> bool CheckOverflows()
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| to | Input| Destination data type for converting aclScalar to determine whether overflow occurs.|

## Returns

If overflow occurs, `true` is returned. Otherwise, `false` is returned.

## Restrictions

None

## Examples

```cpp
// Check whether overflow occurs when the input is converted to fp16 or int16.
void Func(const aclScalar *input) {
    if (input->CheckOverflows<fp16_t>()) {
        return;
    }
    if (input->CheckOverflows<int16_t>()) {
        return;
    }
}
```
