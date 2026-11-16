# aclGetViewOffset

## Function

Obtains the ViewOffset of the aclTensor, that is, the offset relative to ViewShape. The aclTensor is created by the [aclCreateTensor](aclCreateTensor.md) call.

## Prototype

```cpp
aclnnStatus aclGetViewOffset(const aclTensor *tensor, int64_t *offset)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| Input aclTensor. You need to call the [aclCreateTensor](aclCreateTensor.md) API to create an aclTensor in advance.|
| offset | Output| Returned offset value.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

Possible causes:

- If error code 161001 is returned, the `tensor` or `offset` parameter is a null pointer.

## Restrictions

None

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
