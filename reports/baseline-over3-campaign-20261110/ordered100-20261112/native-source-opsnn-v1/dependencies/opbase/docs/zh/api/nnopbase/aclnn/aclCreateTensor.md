# aclCreateTensor

## 功能说明

根据Tensor的数据类型、数据排布格式、维度、步长、偏移、Device侧存储地址等数据，创建aclTensor对象，作为单算子API执行接口的入参。

aclTensor是框架定义的一种用来管理和存储张量数据的结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```cpp
aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storageDims, uint64_t storageDimsNum, void *tensorData)
```

## 参数说明

> 关于aclTensor的StorageShape和ViewShape：
>
> - ViewShape表示Tensor的逻辑shape，是Tensor在实际使用时需要用到的大小。
> - StorageShape表示Tensor的实际物理排布shape，是Tensor在内存上实际存在的大小。
> 举例如下：
> - StorageShape为\[10, 20\]：表示该Tensor在内存上是按照\[10, 20\]排布的。
> - ViewShape为\[2, 5, 20\]：在算子使用时，表示该Tensor可被视为一块\[2, 5, 20\]的数据使用。

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| viewDims | 输入 | tensor的ViewShape维度值，为非负整数。 |
| viewDimsNum | 输入 | tensor的ViewShape维度数。 |
| dataType | 输入 | tensor的数据类型。 |
| stride | 输入 | tensor各维度元素的访问步长，为非负整数。 |
| offset | 输入 | tensor首元素相对于storage的偏移，为非负整数。 |
| format | 输入 | tensor的数据排布格式。 |
| storageDims | 输入 | tensor的StorageShape维度值，为非负整数。 |
| storageDimsNum | 输入 | tensor的StorageShape维度数。 |
| tensorData | 输入 | tensor在Device侧的存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。 |

## 返回值说明

成功则返回创建好的aclTensor，否则返回nullptr。

## 约束说明

- 本接口需与[aclDestroyTensor](aclDestroyTensor.md)接口配套使用，分别完成aclTensor的创建与销毁。
- 如需创建多个aclTensor对象，可调用[aclCreateTensorList](aclCreateTensorList.md)接口来存储张量列表。
- 调用[aclGetDataType](aclGetDataType.md)接口可以获取aclTensor的DataType。
- 调用[aclGetFormat](aclGetFormat.md)接口可以获取aclTensor的format。
- 调用[aclGetStorageShape](aclGetStorageShape.md)接口可以获取aclTensor的StorageShape。
- 调用[aclGetViewOffset](aclGetViewOffset.md)接口可以获取aclTensor的ViewOffset，即ViewShape对应的offset。
- 调用[aclGetViewShape](aclGetViewShape.md)接口可以获取aclTensor的ViewShape。
- 调用[aclGetViewStrides](aclGetViewStrides.md)接口可以获取aclTensor的ViewStrides，即ViewShape对应的stride。
- 调用[aclInitTensor](aclInitTensor.md)接口初始化给定tensor的参数。
- 调用如下接口可刷新或获取不同场景下aclTensor中记录的Device内存地址。
  - [aclSetInputTensorAddr](aclSetInputTensorAddr.md)
  - [aclSetOutputTensorAddr](aclSetOutputTensorAddr.md)
  - [aclSetTensorAddr](aclSetTensorAddr.md)
  - [aclGetRawTensorAddr](aclGetRawTensorAddr.md)
  - [aclSetRawTensorAddr](aclSetRawTensorAddr.md)

## 调用示例

aclTensor的定义与[torch.Tensor](https://pytorch.org/docs/stable/tensors.html)相似，由一块连续或非连续的内存地址和一系列描述信息（如stride、offset等）组成。Tensor根据shape、stride、offset信息，可自由取出内存中的数据，也可获得非连续的内存（如图中y）。

**图 1**  tensor逻辑结构

![](../../../figures/aclTensor.png "tensor逻辑结构")

- 以上图为例，x tensor创建过程如下：

    ```cpp
    aclTensor *CreateXTensor()
    {    
        std::vector<int64_t> viewDims = {2, 4};
        std::vector<int64_t> stride = {4, 1}; // 第1维步长4，第2维步长1
        std::vector<int64_t> storageDims = {2, 4};       
        return aclCreateTensor(viewDims.data(), 2, ACL_FLOAT16, stride.data(), 0, ACL_FORMAT_ND, storageDims.data(), 2, nullptr);
    }
    ```

- 以上图为例，x对应的转置x^T tensor创建过程如下：

    ```cpp
    aclTensor *CreateXTransposedTensor()
    {
        std::vector<int64_t> viewDims = {4, 2};       
        std::vector<int64_t> stride = {1, 4};         // 转置跨度通过stride表示
        std::vector<int64_t> storageDims = {2, 4};    
        return aclCreateTensor(viewDims.data(), 2, ACL_FLOAT16, stride.data(), 0, ACL_FORMAT_ND, storageDims.data(), 2, nullptr);
    }
    ```

基于上述举例，将aclTensor作为单算子API执行接口入参的示例代码如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建aclTensor
aclTensor *xTensor = CreateXTensor();
aclTensor *xTransposedTensor = CreateXTransposedTensor();
// aclTensor作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(xTensor, xTransposedTensor, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclTensor
ret = aclDestroyTensor(xTensor);
ret = aclDestroyTensor(xTransposedTensor);
```
