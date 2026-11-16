# AllocIntArray

## Function

Allocates an `aclIntArray` and initializes it with values from the specified memory.

## Prototype

```cpp
aclIntArray *AllocIntArray(const int64_t *value, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Source data, which is used to initialize `aclIntArray`.|
| size | Input| Number of elements in the source data.|

## Returns

Success: allocated `aclIntArray`. Failure: `nullptr`.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Allocate an aclIntArray with a length of 10.
void Func(aclOpExecutor *executor) {
    int64_t myArray[10];
    aclIntArray *array = executor->AllocIntArray(myArray, 10);
}
```
