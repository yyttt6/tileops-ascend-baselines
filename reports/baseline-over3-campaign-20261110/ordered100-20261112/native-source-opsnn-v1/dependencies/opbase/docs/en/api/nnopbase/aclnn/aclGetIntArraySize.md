# aclGetIntArraySize

## Function

Obtains the size of the aclIntArray created by calling the [aclCreateIntArray](aclCreateIntArray.md) API.

## Prototype

```cpp
aclnnStatus aclGetIntArraySize(const aclIntArray *array, uint64_t *size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| Input aclIntArray.|
| size | Output| Size of the aclIntArray.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `array` or `size` is a null pointer.

## Restrictions

None

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclIntArray.
std::vector<int64_t> valueData = {1, 1, 2, 3};
aclIntArray *valueArray = aclCreateIntArray(valueData.data(), valueData.size());
...
// Obtain the size of valueArray by calling aclGetIntArraySize.
uint64_t size = 0;
auto ret = aclGetIntArraySize(valueArray, &size); // The obtained size of the valueArray is 4.
...
// Destroy the aclIntArray.
ret = aclDestroyIntArray(valueArray);
```
