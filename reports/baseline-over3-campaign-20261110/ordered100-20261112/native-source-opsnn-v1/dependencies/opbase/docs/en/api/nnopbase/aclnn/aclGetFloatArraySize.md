# aclGetFloatArraySize

## Function

Obtains the size of the aclFloatArray created by calling the [aclCreateFloatArray](aclCreateFloatArray.md) API.

## Prototype

```cpp
aclnnStatus aclGetFloatArraySize(const aclFloatArray *array, uint64_t *size)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| array | Input| Input aclFloatArray.|
| size | Output| Returned size of the aclFloatArray.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, `array` or `size` is a null pointer.

## Restrictions

None

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// Create an aclFloatArray.
std::vector<float> scalesData = {1.0, 1.0, 2.0, 2.0};
aclFloatArray *scales = aclCreateFloatArray(scalesData.data(), scalesData.size());
...
// Obtain the size of scales by calling aclGetFloatArraySize.
uint64_t size = 0;
auto ret = aclGetFloatArraySize(scales, &size); // The obtained size of the scales is 4.
...
// Destroy the aclFloatArray.
ret = aclDestroyFloatArray(scales);
```
