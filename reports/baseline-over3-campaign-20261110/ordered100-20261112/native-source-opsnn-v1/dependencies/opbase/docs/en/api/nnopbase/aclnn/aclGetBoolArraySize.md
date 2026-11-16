# aclGetBoolArraySize

## Function

Obtains the size of the aclBoolArray created by calling the [aclCreateBoolArray](aclCreateBoolArray.md) API.

## Prototype

```cpp
aclnnStatus aclGetBoolArraySize(const aclBoolArray *array, uint64_t *size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| Input aclBoolArray.|
| size | Output| Size of the aclBoolArray.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `array` or `size` is a null pointer.

## Restrictions

None

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclBoolArray.
std::vector<bool> maskData = {true, false};
aclBoolArray *mask = aclCreateBoolArray(maskData.data(), maskData.size());
...
// Obtain the mask size by calling aclGetBoolArraySize.
uint64_t size = 0;
auto ret = aclGetBoolArraySize(mask, &size);     // The obtained mask size is 2.
...
// Destroy the aclBoolArray.
ret = aclDestroyBoolArray(mask);
```
