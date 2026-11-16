# aclSetAclOpExecutorRepeatable

## 功能说明

开启aclOpExecutor为可复用状态。当用户想复用已有的aclOpExecutor时，必须在一阶段接口aclxxXxxGetWorkspaceSize运行完成后，立即使用该接口开启复用，后续可多次调用第二段接口aclxxXxx进行算子执行。

aclOpExecutor是框架定义的算子执行器，用于执行算子计算的容器，开发者无需关注其内部实现。

## 函数原型

```cpp
aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor *executor)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| executor | 输入 | 待设置复用的aclOpExecutor。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回561103：executor是空指针。

## 约束说明

- 目前采用AI CPU和AI Core计算单元的算子支持开启aclOpExecutor可复用。
- 调用单算子API执行接口时，如下场景无法开启aclOpExecutor复用：

  - 如果使用了HostToDevice、DeviceToDevice拷贝相关的L0层API，如CopyToNpu、CopyNpuToNpu、CopyToNpuSync等，不支持aclOpExecutor复用。
  - 如果使用了L0层ViewCopy接口，同时ViewCopy的源地址和目的地址相同时，不支持aclOpExecutor复用。

- 调用单算子API执行接口时，不允许算子API内部创建Device Tensor，只允许使用外部传入的Tensor。
- 设置成复用状态的aclOpExecutor在第二段接口执行完后不会对executor的资源进行清理，需要和[aclDestroyAclOpExecutor](aclDestroyAclOpExecutor.md)配套使用清理资源。

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
aclTensor *output = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list[] = {tensor1, tensor2};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
// AddCustom算子有两个输入（aclTensorList和aclTensor），一个输出（aclTensor）
// 调用第1段接口
aclnnAddCustomGetWorkspaceSize(tensorList, tensor3, output, &workspaceSize, &executor);
// 设置executor为可复用
aclSetAclOpExecutorRepeatable(executor);  
void *addr;
aclSetDynamicInputTensorAddr(executor, 0, 0, tensorList, addr);   // 刷新输入tensorlist中第1个aclTensor的device地址
aclSetDynamicInputTensorAddr(executor, 0, 1, tensorList, addr);  // 刷新输入tensorlist中第2个aclTensor的device地址
...
// 调用第2段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);
// 清理executor
aclDestroyAclOpExecutor(executor);  
```
