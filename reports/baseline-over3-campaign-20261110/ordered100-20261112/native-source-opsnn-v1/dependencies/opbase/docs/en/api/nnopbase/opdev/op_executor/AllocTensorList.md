# AllocTensorList

## Function

Allocates an aclTensorList and specifies the aclTensors contained in it.

## Prototype

```cpp
aclTensorList *AllocTensorList(const aclTensor *const *tensors, uint64_t size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensors | Input| Source data, which is used to initialize aclTensorList.|
| size | Input| Number of elements in the source data.|

## Returns

Success: allocated aclTensorList object. Failure: nullptr.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Initialize five aclTensors and encapsulate them into an aclTensorList.
void Func(aclOpExecutor *executor) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    std::vector<aclTensor *> tensors;
    for (int64_t i = 1; i <= 5; i++) {
        aclTensor *tensor = executor->AllocTensor(newShape, DT_INT64, ge::FORMAT_ND);
        tensors.push_back(tensor);
    }
    aclTensorList *tensorList = executor->AllocTensorList(tensors.data(), tensors.size());
}
```
