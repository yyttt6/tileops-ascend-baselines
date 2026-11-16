# aclInitTensor

## 功能说明

初始化给定aclTensor的参数，aclTensor由[aclCreateTensor](aclCreateTensor.md)接口创建。

当用户想复用已创建aclTensor时，可使用该接口先重置aclTensor的各项属性。

## 函数原型

```cpp
aclnnStatus aclInitTensor(aclTensor *tensor, const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storageDims, uint64_t storageDimsNum, void *tensorDataAddr)
```

## 参数说明

> 关于aclTensor的StorageShape和ViewShape说明：
>
> - ViewShape表示Tensor的逻辑shape，是Tensor在实际使用时需要用到的大小。
> - StorageShape表示Tensor的实际物理排布shape，是Tensor在内存上实际存在的大小。
> 举例如下：
> - StorageShape为\[10, 20\]：表示该Tensor在内存上是按照\[10, 20\]排布的。
> - ViewShape为\[2, 5, 20\]：在算子使用时，表示该Tensor可被视为一块\[2, 5, 20\]的数据使用。

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输入 | 待初始化参数的aclTensor。 |
| viewDims | 输入 | tensor的ViewShape维度值，为非负整数。 |
| viewDimsNum | 输入 | tensor的ViewShape维度数。 |
| dataType | 输入 | tensor的数据类型。 |
| stride | 输入 | tensor各维度元素的访问步长，为非负整数。 |
| offset | 输入 | tensor首元素相对于storage的偏移，为非负整数。 |
| format | 输入 | tensor的数据排布格式。 |
| storageDims | 输入 | tensor的StorageShape维度值，为非负整数。 |
| storageDimsNum | 输入 | tensor的StorageShape维度数。 |
| tensorDataAddr | 输入 | tensor在Device侧的存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
std::vector<int64_t> viewDims = {2, 4};
std::vector<int64_t> stride = {4, 1};
std::vector<int64_t> storageDims = {2, 4};  
// tensor为复用的已经创建的aclTensor
// deviceAddr为tensor在Device侧的存储地址
auto ret = aclInitTensor(tensor, viewDims.data(), viewDims.size(), ACL_FLOAT16, stride.data(), 0, aclFormat::ACL_FORMAT_ND, storageDims.data(), storageDims.size(), deviceAddr);
```
