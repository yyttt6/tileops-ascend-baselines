# 公共接口列表

本章是调用CANN API过程中依赖的公共Meta接口，如创建/释放aclTensor、aclScalar、aclIntArray等。

**头文件说明**：调用本章接口时，请按实际情况include依赖的头文件，一般定义在`${INSTALL_DIR}/include`目录。其中\$\{INSTALL\_DIR\}表示为CANN软件安装后的路径，以root安装举例，安装后路径为`/usr/local/Ascend/cann`。

| 接口 | 说明 | 所属头文件 |
| --- | --- | --- |
| [aclCreateBoolArray](aclCreateBoolArray.md) | 创建aclBoolArray。 | aclnn/acl_meta.h |
| [aclCreateFloatArray](aclCreateFloatArray.md) | 创建aclFloatArray。 | aclnn/acl_meta.h |
| [aclCreateIntArray](aclCreateIntArray.md) | 创建aclIntArray。 | aclnn/acl_meta.h |
| [aclCreateScalar](aclCreateScalar.md) | 创建aclScalar。 | aclnn/acl_meta.h |
| [aclCreateScalarList](aclCreateScalarList.md) | 创建aclScalarList。 | aclnn/acl_meta.h |
| [aclCreateTensor](aclCreateTensor.md) | 创建aclTensor。 | aclnn/acl_meta.h |
| [aclCreateTensorList](aclCreateTensorList.md) | 创建aclTensorList。 | aclnn/acl_meta.h |
| [aclDestroyAclOpExecutor](aclDestroyAclOpExecutor.md) | 销毁可复用状态的aclOpExecutor。 | aclnn/acl_meta.h |
| [aclDestroyBoolArray](aclDestroyBoolArray.md) | 销毁创建的aclBoolArray。 | aclnn/acl_meta.h |
| [aclDestroyFloatArray](aclDestroyFloatArray.md) | 销毁创建的aclFloatArray。 | aclnn/acl_meta.h |
| [aclDestroyIntArray](aclDestroyIntArray.md) | 销毁创建的aclIntArray。 | aclnn/acl_meta.h |
| [aclDestroyScalar](aclDestroyScalar.md) | 销毁创建的aclScalar。 | aclnn/acl_meta.h |
| [aclDestroyScalarList](aclDestroyScalarList.md) | 销毁创建的aclScalarList，对于aclScalarList内的Scalar不需要再重复释放。 | aclnn/acl_meta.h |
| [aclDestroyTensor](aclDestroyTensor.md) | 销毁创建的aclTensor。 | aclnn/acl_meta.h |
| [aclDestroyTensorList](aclDestroyTensorList.md) | 销毁创建的aclTensorList，对于aclTensorList内的Tensor不需要再重复释放。 | aclnn/acl_meta.h |
| [aclGetBoolArraySize](aclGetBoolArraySize.md) | 获取aclBoolArray的大小。 | aclnn/acl_meta.h |
| [aclGetDataType](aclGetDataType.md) | 获取aclTensor的DataType。 | aclnn/acl_meta.h |
| [aclGetFloatArraySize](aclGetFloatArraySize.md) | 获取aclFloatArray的大小。 | aclnn/acl_meta.h |
| [aclGetFormat](aclGetFormat.md) | 获取aclTensor的format。 | aclnn/acl_meta.h |
| [aclGetIntArraySize](aclGetIntArraySize.md) | 获取aclIntArray的大小。 | aclnn/acl_meta.h |
| [aclGetRawTensorAddr](aclGetRawTensorAddr.md) | 获取aclTensor中原始记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclGetScalarListSize](aclGetScalarListSize.md) | 获取aclScalarList的大小。 | aclnn/acl_meta.h |
| [aclGetStorageShape](aclGetStorageShape.md) | 获取aclTensor的StorageShape。 | aclnn/acl_meta.h |
| [aclGetTensorListSize](aclGetTensorListSize.md) | 获取aclTensorList的大小。 | aclnn/acl_meta.h |
| [aclGetViewOffset](aclGetViewOffset.md) | 获取aclTensor的ViewOffset，即ViewShape对应的offset。 | aclnn/acl_meta.h |
| [aclGetViewShape](aclGetViewShape.md) | 获取aclTensor的ViewShape。 | aclnn/acl_meta.h |
| [aclGetViewStrides](aclGetViewStrides.md) | 获取aclTensor的ViewStrides，即ViewShape对应的stride。 | aclnn/acl_meta.h |
| [aclInitTensor](aclInitTensor.md) | 初始化给定tensor的参数。 | aclnn/acl_meta.h |
| [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) | 开启aclOpExecutor为可复用状态。 | aclnn/acl_meta.h |
| [aclSetDynamicInputTensorAddr](aclSetDynamicInputTensorAddr.md) | 开启aclOpExecutor可复用后，若输入Device内存地址变更，需要刷新输入aclTensorList中记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclSetDynamicOutputTensorAddr](aclSetDynamicOutputTensorAddr.md) | 开启aclOpExecutor可复用后，若输出Device内存地址变更，需要刷新输出aclTensorList中记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclSetDynamicTensorAddr](aclSetDynamicTensorAddr.md) | 开启aclOpExecutor可复用后，若输入或输出Device内存地址变更，需要刷新对应aclTensorList中记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclSetInputTensorAddr](aclSetInputTensorAddr.md) | 开启aclOpExecutor可复用后，若输入Device内存地址变更，需要刷新输入aclTensor中记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclSetOutputTensorAddr](aclSetOutputTensorAddr.md) | 开启aclOpExecutor可复用后，若输出Device内存地址变更，需要刷新输出aclTensor中记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclSetRawTensorAddr](aclSetRawTensorAddr.md) | 刷新aclTensor中原始记录的Device内存地址。 | aclnn/acl_meta.h |
| [aclSetTensorAddr](aclSetTensorAddr.md) | 开启aclOpExecutor可复用后，若输入或输出Device内存地址变更，需要刷新对应aclTensor中记录的Device内存地址。 | aclnn/acl_meta.h |
| AclSetInputTensorAddr | [预留接口](reserved_interface.md)，开发者无需关注。 | aclnn/acl_meta.h |
| AclSetOutputTensorAddr | [预留接口](reserved_interface.md)，开发者无需关注。 | aclnn/acl_meta.h |
| AclSetDynamicInputTensorAddr | [预留接口](reserved_interface.md)，开发者无需关注。 | aclnn/acl_meta.h |
| AclSetDynamicOutputTensorAddr | [预留接口](reserved_interface.md)，开发者无需关注。 | aclnn/acl_meta.h |
| AclSetTensorAddr | [预留接口](reserved_interface.md)，开发者无需关注。 | aclnn/acl_meta.h |
| AclSetDynamicTensorAddr | [预留接口](reserved_interface.md)，开发者无需关注。 | aclnn/acl_meta.h |
| [aclnnInit](aclnnInit.md) | aclnn API的初始化函数。 | aclnn/aclnn_base.h |
| [aclnnFinalize](aclnnFinalize.md) | aclnn API的去初始化函数。 | aclnn/aclnn_base.h |
