# AllocScalarList

## Function

Allocates an aclScalarList and specifies the aclScalars contained in it.

## Prototype

```cpp
aclScalarList *AllocScalarList(const aclScalar *const *scalars, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| scalars | Input| Source data, which is used to initialize aclScalarList.|
| size | Input| Number of elements in the source data.|

## Returns

Success: allocated aclScalarList object. Failure: nullptr.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Initialize five aclScalars and encapsulate them into an aclScalarList.
void Func(aclOpExecutor *executor) {
    int64_t val = 5;
    std::vector<aclScalar *> scalars;
    for (int64_t i = 1; i <= 5; i++) {
        aclScalar *scalar = executor->AllocScalar(val);
        scalars.push_back(scalar);
    }
    aclScalarList *scalarList = executor->AllocScalarList(scalars.data(), scalars.size());
}
```
