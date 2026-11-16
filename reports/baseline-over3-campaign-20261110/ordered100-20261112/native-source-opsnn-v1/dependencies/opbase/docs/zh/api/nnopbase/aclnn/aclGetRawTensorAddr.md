# aclGetRawTensorAddr

## 功能说明

获取aclTensor中原始记录的Device内存地址，aclTensor由[aclCreateTensor](aclCreateTensor.md)接口创建。

## 函数原型

```cpp
aclnnStatus aclGetRawTensorAddr(const aclTensor *tensor, void **addr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输入 | 输入的aclTensor指针。 |
| addr | 输出 | 返回的aclTensor中记录的Device内存地址。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

可能失败的原因：

- 返回161001：参数tensor或者addr为空指针。

## 约束说明

- 必须在一阶段接口aclxxXxxGetWorkspaceSize之前或者二阶段接口aclxxXxx之后使用，不支持在一阶段与二阶段接口之间使用。
- 本接口可与[aclSetRawTensorAddr](aclSetRawTensorAddr.md)接口配套使用，查看刷新后的结果是否符合预期。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建输入和输出张量inputTensor和outputTensor
std::vector<int64_t> shape = {1, 2, 3};
void *addr1;
void *addr2;
... // 申请device内存addr1， addr2
aclTensor *inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr1);
aclTensor *outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr2);

void *getAddr1 = nullptr;
void *getAddr2 = nullptr;
// 获取inputTensor中记录的device内存地址，此处获取到的指针getAddr1指向的内存地址与addr1一致
auto ret = aclGetRawTensorAddr(inputTensor, &getAddr1);
// 获取outputTensor中记录的device内存地址，此处获取到的指针getAddr2指向的内存地址与addr2一致
ret = aclGetRawTensorAddr(outputTensor, &getAddr2);

// 调用Xxx算子一、二阶段接口
ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...
```
