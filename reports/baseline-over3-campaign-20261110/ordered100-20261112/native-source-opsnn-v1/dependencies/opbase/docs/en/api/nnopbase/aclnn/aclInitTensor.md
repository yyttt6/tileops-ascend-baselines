# aclInitTensor

## Function

Initializes the parameters of a given aclTensor created by the [aclCreateTensor](aclCreateTensor.md) call.

If you want to reuse an existing aclTensor, you can call this API to reset the aclTensor attributes.

## Prototype

```cpp
aclnnStatus aclInitTensor(aclTensor *tensor, const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storageDims, uint64_t storageDimsNum, void *tensorDataAddr)
```

## Parameters

> StorageShape and ViewShape of aclTensor:
>
> - ViewShape indicates the logical shape of the tensor, which is the size of the tensor required for actual use.
> - StorageShape indicates the actual physical layout shape of the tensor, which is the actual size of the tensor in the memory.
> The following is an example:
> - If StorageShape is \[10, 20\], the tensor is arranged in the memory based on \[10, 20\].
> - If ViewShape is \[2, 5, 20\], the tensor can be considered as a data block \[2, 5, 20\] for operator use.

| Parameter| Input/Output| Description|
| --- | --- | --- |
| tensor | Input| aclTensor whose parameters are to be initialized.|
| viewDims | Input| ViewShape dimension value of a tensor, which is a non-negative integer.|
| viewDimsNum | Input| ViewShape dimension number of a tensor.|
| dataType | Input| Data type of a tensor.|
| stride | Input| Access stride of elements in each dimension of the tensor, which is a non-negative integer.|
| offset | Input| Offset of the first element of the tensor relative to storage, which is a non-negative integer.|
| format | Input| Tensor format.|
| storageDims | Input| StorageShape dimension value of the tensor, which is a non-negative integer.|
| storageDimsNum | Input| StorageShape dimension number of a tensor.|
| tensorDataAddr | Input| Storage address of the tensor on the device. The address must be 32-byte aligned. Otherwise, an undefined error may occur.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

None

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

```cpp
std::vector<int64_t> viewDims = {2, 4};
std::vector<int64_t> stride = {4, 1};
std::vector<int64_t> storageDims = {2, 4};  
// The created aclTensor is reused as a tensor.
// deviceAddr indicates the storage address of the tensor on the device.
auto ret = aclInitTensor(tensor, viewDims.data(), viewDims.size(), ACL_FLOAT16, stride.data(), 0, aclFormat::ACL_FORMAT_ND, storageDims.data(), storageDims.size(), deviceAddr);
```
