# aclGetTensorListSize

## Function

Obtains the size of the aclTensorList created by calling the [aclCreateTensorList](aclCreateTensorList.md) API.

## Prototype

```cpp
aclnnStatus aclGetTensorListSize(const aclTensorList *tensorList, uint64_t *size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensorList | Input| Input aclTensorList.|
| size | Output| Size of the aclTensorList.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `tensorList` or `size` is a null pointer.

## Restrictions

None

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclTensorList.
std::vector<aclTensor *> tmp{input1, input2};
aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
...
// Obtain the size of the tensorList.
uint64_t size = 0;
auto ret = aclGetTensorListSize(tensorList, &size); // The obtained size of the tensorList is 2.
...
// Destroy the aclTensorList.
ret = aclDestroyTensorList(tensorList);
```
