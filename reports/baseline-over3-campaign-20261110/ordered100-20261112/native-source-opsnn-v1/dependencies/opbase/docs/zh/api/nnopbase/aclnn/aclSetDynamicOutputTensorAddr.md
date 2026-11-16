# aclSetDynamicOutputTensorAddr

## 功能说明

通过[aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md)开启aclOpExecutor可复用后，若**输出Device内存地址变更**，需要刷新**输出aclTensorList**中记录的Device内存地址。

## 函数原型

```cpp
aclnnStatus aclSetDynamicOutputTensorAddr(aclOpExecutor *executor, size_t irIndex, const size_t relativeIndex, aclTensorList *tensors, void *addr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| executor | 输入 | 设置为复用状态的aclOpExecutor。 |
| irIndex | 输入 | 待刷新的aclTensorList在算子IR原型定义中的索引，从0开始计数。 |
| relativeIndex | 输入 | 待刷新的aclTensor在aclTensorList中的索引。如果aclTensorList有N个Tensor，其取值范围为[0, N-1]。 |
| tensors | 输入 | 待刷新的aclTensorList指针。 |
| addr | 输入 | 需要刷新到指定aclTensor中的Device存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回561103：executor或tensors是空指针。
- 返回161002：relativeIndex\>=tensors里tensor的个数。
- 返回161002：irIndex\>=算子原型输出参数的个数。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建输入和输出的aclTensor和aclTensorList
std::vector<int64_t> shape = {1, 2, 3};
aclTensor *tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *tensor3 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *tensor4 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *output = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list[] = {tensor3, tensor4};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
// AddCustom算子有两个输入（aclTensor），一个输出（aclTensorList）
// 调用第1段接口
aclnnAddCustomGetWorkspaceSize(tensor1, tensor2, tensorList , &workspaceSize, &executor);
// 设置executor为可复用
aclSetAclOpExecutorRepeatable(executor);  
void *addr;
aclSetDynamicOutputTensorAddr(executor, 0, 0, tensorList, addr); // 刷新输出tensorlist中第1个aclTensor的device地址
aclSetDynamicOutputTensorAddr(executor, 0, 1, tensorList, addr); // 刷新输出tensorlist中第2个aclTensor的device地址
...
 // 调用第2段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);
 // 清理executor
aclDestroyAclOpExecutor(executor); 
```
