# Common APIs

This section describes the common meta APIs required for calling CANN APIs, such as creating and destroying aclTensor, aclScalar, and aclIntArray.

Header file description: When calling the APIs in this section, include the dependent header files based on the site requirements. The header files are stored in the *`$\{INSTALL\_DIR\}*/include` directory. Replace *`$\{INSTALL\_DIR\}`* with the CANN software installation path. For example, if the `Ascend-cann-toolkit` software package is installed as the `root` user, the file storage path after the installation is `/usr/local/Ascend/cann`.

| API| Description| Header File|
| --- | --- | --- |
| [aclCreateBoolArray](aclCreateBoolArray.md) | Creates an aclBoolArray.| aclnn/acl_meta.h |
| [aclCreateFloatArray](aclCreateFloatArray.md) | Creates an aclFloatArray.| aclnn/acl_meta.h |
| [aclCreateIntArray](aclCreateIntArray.md) | Creates an aclIntArray.| aclnn/acl_meta.h |
| [aclCreateScalar](aclCreateScalar.md) | Creates an aclScalar.| aclnn/acl_meta.h |
| [aclCreateScalarList](aclCreateScalarList.md) | Creates an aclScalarList.| aclnn/acl_meta.h |
| [aclCreateTensor](aclCreateTensor.md) | Creates an aclTensor.| aclnn/acl_meta.h |
| [aclCreateTensorList](aclCreateTensorList.md) | Creates an aclTensorList.| aclnn/acl_meta.h |
| [aclDestroyAclOpExecutor](aclDestroyAclOpExecutor.md) | Destroys a reusable aclOpExecutor.| aclnn/acl_meta.h |
| [aclDestroyBoolArray](aclDestroyBoolArray.md) | Destroys the created aclBoolArray.| aclnn/acl_meta.h |
| [aclDestroyFloatArray](aclDestroyFloatArray.md) | Destroys the created aclFloatArray.| aclnn/acl_meta.h |
| [aclDestroyIntArray](aclDestroyIntArray.md) | Destroys the created aclIntArray.| aclnn/acl_meta.h |
| [aclDestroyScalar](aclDestroyScalar.md) | Destroys the created aclScalar.| aclnn/acl_meta.h |
| [aclDestroyScalarList](aclDestroyScalarList.md) | Destroys the created aclScalarList. The scalars in the aclScalarList do not need to be destroyed again.| aclnn/acl_meta.h |
| [aclDestroyTensor](aclDestroyTensor.md) | Destroys the created aclTensor.| aclnn/acl_meta.h |
| [aclDestroyTensorList](aclDestroyTensorList.md) | Destroys the created aclTensorList. The tensors in the aclTensorList do not need to be destroyed again.| aclnn/acl_meta.h |
| [aclGetBoolArraySize](aclGetBoolArraySize.md) | Obtains the size of the aclBoolArray.| aclnn/acl_meta.h |
| [aclGetDataType](aclGetDataType.md) | Obtains the data type of the aclTensor.| aclnn/acl_meta.h |
| [aclGetFloatArraySize](aclGetFloatArraySize.md) | Obtains the size of the aclFloatArray.| aclnn/acl_meta.h |
| [aclGetFormat](aclGetFormat.md) | Obtains the format of the aclTensor.| aclnn/acl_meta.h |
| [aclGetIntArraySize](aclGetIntArraySize.md) | Obtains the size of the aclIntArray.| aclnn/acl_meta.h |
| [aclGetRawTensorAddr](aclGetRawTensorAddr.md) | Obtains the device memory address originally recorded in the aclTensor.| aclnn/acl_meta.h |
| [aclGetScalarListSize](aclGetScalarListSize.md) | Obtains the size of an aclScalarList.| aclnn/acl_meta.h |
| [aclGetStorageShape](aclGetStorageShape.md) | Obtains the StorageShape of an aclTensor.| aclnn/acl_meta.h |
| [aclGetTensorListSize](aclGetTensorListSize.md) | Obtains the size of an aclTensorList.| aclnn/acl_meta.h |
| [aclGetViewOffset](aclGetViewOffset.md) | Obtains the ViewOffset of the aclTensor, that is, the offset corresponding to ViewShape.| aclnn/acl_meta.h |
| [aclGetViewShape](aclGetViewShape.md) | Obtains the ViewShape of an aclTensor.| aclnn/acl_meta.h |
| [aclGetViewStrides](aclGetViewStrides.md) | Obtains ViewStrides of an aclTensor, that is, the stride corresponding to ViewShape.| aclnn/acl_meta.h |
| [aclInitTensor](aclInitTensor.md) | Initializes the parameters of a given tensor.| aclnn/acl_meta.h |
| [aclSetAclOpExecutorRepeatable](aclSetAclOpExecutorRepeatable.md) | Enables aclOpExecutor to be reusable.| aclnn/acl_meta.h |
| [aclSetDynamicInputTensorAddr](aclSetDynamicInputTensorAddr.md) | After aclOpExecutor reuse is enabled, if the input device memory address changes, the device memory address recorded in the input aclTensorList needs to be updated.| aclnn/acl_meta.h |
| [aclSetDynamicOutputTensorAddr](aclSetDynamicOutputTensorAddr.md) | After aclOpExecutor reuse is enabled, if the output device memory address changes, the device memory address recorded in the output aclTensorList needs to be updated.| aclnn/acl_meta.h |
| [aclSetDynamicTensorAddr](aclSetDynamicTensorAddr.md) | After aclOpExecutor reuse is enabled, if the input or output device memory address changes, the device memory address recorded in the corresponding aclTensorList needs to be updated.| aclnn/acl_meta.h |
| [aclSetInputTensorAddr](aclSetInputTensorAddr.md) | After aclOpExecutor reuse is enabled, if the input device memory address changes, the device memory address recorded in the input aclTensor needs to be updated.| aclnn/acl_meta.h |
| [aclSetOutputTensorAddr](aclSetOutputTensorAddr.md) | After aclOpExecutor reuse is enabled, if the output device memory address changes, the device memory address recorded in the output aclTensor needs to be updated.| aclnn/acl_meta.h |
| [aclSetRawTensorAddr](aclSetRawTensorAddr.md) | Updates the device memory address originally recorded in the aclTensor.| aclnn/acl_meta.h |
| [aclSetTensorAddr](aclSetTensorAddr.md) | After aclOpExecutor reuse is enabled, if the input or output device memory address changes, the device memory address recorded in the corresponding aclTensor needs to be updated.| aclnn/acl_meta.h |
| AclSetInputTensorAddr | [Reserved APIs](reserved_apis.md) can be ignored.| aclnn/acl_meta.h |
| AclSetOutputTensorAddr | [Reserved APIs](reserved_apis.md) can be ignored.| aclnn/acl_meta.h |
| AclSetDynamicInputTensorAddr | [Reserved APIs](reserved_apis.md) can be ignored.| aclnn/acl_meta.h |
| AclSetDynamicOutputTensorAddr | [Reserved APIs](reserved_apis.md) can be ignored.| aclnn/acl_meta.h |
| AclSetTensorAddr | [Reserved APIs](reserved_apis.md) can be ignored.| aclnn/acl_meta.h |
| AclSetDynamicTensorAddr | [Reserved APIs](reserved_apis.md) can be ignored.| aclnn/acl_meta.h |
| [aclnnInit](aclnnInit.md) | Initialization function of the aclnn API.| aclnn/aclnn_base.h |
| [aclnnFinalize](aclnnFinalize.md) | Deinitialization function of the aclnn API.| aclnn/aclnn_base.h |
