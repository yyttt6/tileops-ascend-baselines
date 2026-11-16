# 预留接口

本章罗列了预留的公共接口，后续版本存在变更或废弃的情况，不建议开发者使用，开发者无需关注。

**表 1**  预留接口表

| 接口名称 | 接口说明 |
| --- | --- |
| AclSetInputTensorAddr(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr) | 功能同[aclSetInputTensorAddr](aclSetInputTensorAddr.md)一致。 |
| AclSetOutputTensorAddr(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr) | 功能同[aclSetOutputTensorAddr](aclSetOutputTensorAddr.md)一致。 |
| AclSetDynamicInputTensorAddr(aclOpExecutor *executor, size_t irIndex, const size_t relativeIndex, aclTensorList *tensors, void *addr) | 功能同[aclSetDynamicInputTensorAddr](aclSetDynamicInputTensorAddr.md)一致。 |
| AclSetDynamicOutputTensorAddr(aclOpExecutor *executor, size_t irIndex, const size_t relativeIndex, aclTensorList *tensors, void *addr) | 功能同[aclSetDynamicOutputTensorAddr](aclSetDynamicOutputTensorAddr.md)一致。 |
| AclSetTensorAddr(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr) | 功能同[aclSetTensorAddr](aclSetTensorAddr.md)一致。 |
| AclSetDynamicTensorAddr(aclOpExecutor *executor, size_t irIndex, const size_t relativeIndex, aclTensorList *tensors, void *addr) | 功能同[aclSetDynamicTensorAddr](aclSetDynamicTensorAddr.md)一致。 |
