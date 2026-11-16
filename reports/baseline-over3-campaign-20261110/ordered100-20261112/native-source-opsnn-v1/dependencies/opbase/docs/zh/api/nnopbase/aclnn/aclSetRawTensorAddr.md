# aclSetRawTensorAddr

## 功能说明

刷新aclTensor中原始记录的Device内存地址，aclTensor由[aclCreateTensor](aclCreateTensor.md)接口创建。

通常情况下，若网络需要频繁复用aclTensor（即保持shape、format等属性一致），可使用本接口刷新aclTensor原始的Device内存地址，达到复用的目的。

## 函数原型

```cpp
aclnnStatus aclSetRawTensorAddr(aclTensor *tensor, void *addr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输入 | 待刷新的aclTensor指针。 |
| addr | 输入 | 需要刷新到指定aclTensor中的Device存储地址，该地址必须32字节对齐，否则可能会出现未定义错误。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数tensor为空指针。

## 约束说明

- 必须在一阶段接口aclxxXxxGetWorkspaceSize之前或者二阶段接口aclxxXxx之后使用，不支持在一阶段与二阶段接口之间使用。
- 本接口可与[aclGetRawTensorAddr](aclGetRawTensorAddr.md)接口配套使用，查看刷新后的结果是否符合预期。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建输入和输出张量inputTensor和outputTensor
std::vector<int64_t> shape = {1, 2, 3};
aclTensor *inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
// 调用Xxx算子一、二阶段接口
auto ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...

void *addr1;
void *addr2;
... //申请device内存addr1， addr2
ret = aclSetRawTensorAddr(inputTensor, addr1); // 刷新输入张量inputTensor的device地址
ret = aclSetRawTensorAddr(outputTensor, addr2); // 刷新输出张量outputTensor的device地址
...
// 复用inputTensor，outputTensor后，调用Yyy算子aclnn一、二阶段接口
auto ret = aclnnYyyGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclnnYyy(workspace, workspaceSize, executor, stream);
...
```
