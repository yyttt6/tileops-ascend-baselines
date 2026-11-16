# aclGetStorageShape

## Function

Obtains the StorageShape of the aclTensor created by the [aclCreateTensor](aclCreateTensor.md) call.

StorageShape indicates the actual physical layout shape of the aclTensor, which is the actual size of the tensor in the memory. If StorageShape is \[10, 20\], the layout of the aclTensor in the memory is based on \[10, 20\].

## Prototype

```cpp
aclnnStatus aclGetStorageShape(const aclTensor *tensor, int64_t **storageDims, uint64_t *storageDimsNum)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| Address of the input aclTensor. You need to call the [aclCreateTensor](aclCreateTensor.md) API to create an aclTensor in advance.|
| storageDims | Output| StorageShape dimension value.|
| storageDimsNum | Output| StorageShape dimension number.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, the `tensor`, `storageDims`, or `storageDimsNum` parameter is a null pointer.

## Restrictions

The memory for `storageDims` is internally allocated by this API. After use, you need to manually release it.

## Examples

Assume that there is an aclTensor object (xTensor). Obtain its attributes such as the data type, data layout format, dimension, stride, and offset, and create an aclTensor object (yTensor) based on these attributes.

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
// 1. Create an xTensor.
int64_t xViewDims = {2, 4};       
int64_t xStridesValue = {4, 1};  // The stride of the first dimension is 4, and that of the second dimension is 1. 
int64_t xStorageDims = {2, 4};    
xTensor = aclCreateTensor(xViewDims, 2, ACL_FLOAT16, xStridesValue, 0, ACL_FORMAT_ND, xStorageDims, 2, nullptr);

// 2. Obtain the attribute values of xTensor.
// Obtain the logical shape of xTensor. viewDims is {2, 4}, and viewDimsNum is 2.
int64_t *viewDims = nullptr;
uint64_t viewDimsNum = 0;
auto ret = aclGetViewShape(xTensor, &viewDims, &viewDimsNum);
// Obtain the data type (ACL_FLOAT16) of xTensor.
aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
ret = aclGetDataType(xTensor, &dataType);
// Obtain the stride information about xTensor. stridesValue is {4, 1}, and stridesNum is 2.
int64_t *stridesValue = nullptr;
uint64_t stridesNum = 0;
ret = aclGetViewStrides(xTensor, &stridesValue, &stridesNum);
// Obtain the offset of the first element of xTensor relative to storage. The offset is 0.
int64_t offset = 0;
ret = aclGetViewOffset(xTensor, &offset);
// Obtain the data layout format (ACL_FORMAT_ND) of xTensor.
aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
ret = aclGetFormat(xTensor, &format);
// Obtain the actual physical shape of xTensor. storageDims is {2, 4}, and storageDimsNum is 2.
int64_t *storageDims = nullptr;
uint64_t storageDimsNum = 0;
ret = aclGetStorageShape(xTensor, &storageDims, &storageDimsNum);
// Device address
void *deviceAddr;

// 3. Create a tensor based on the xTensor attributes.
aclTensor *yTensor = aclCreateTensor(viewDims, viewDimsNum, dataType, stridesValue, offset, format, storageDims, storageDimsNum, deviceAddr);

// 4. Manually free memory.
delete[] viewDims;
delete[] stridesValue;
delete[] storageDims;
```
