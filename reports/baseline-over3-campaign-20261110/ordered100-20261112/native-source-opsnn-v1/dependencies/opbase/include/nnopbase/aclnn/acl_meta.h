/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_EXTERNAL_ACL_META_H
#define OP_API_OP_API_COMMON_INC_EXTERNAL_ACL_META_H

#include <cstdint>
#include <cstdlib>
#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ACLNN_META
#define ACLNN_META
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

typedef int32_t aclnnStatus;
constexpr aclnnStatus OK = 0;
#endif

ACL_FUNC_VISIBILITY aclTensor* aclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                               const int64_t* stride, int64_t offset, aclFormat format,
                                               const int64_t* storageDims, uint64_t storageDimsNum, void* tensorData);

ACL_FUNC_VISIBILITY aclScalar* aclCreateScalar(void* value, aclDataType dataType);
ACL_FUNC_VISIBILITY aclIntArray* aclCreateIntArray(const int64_t* value, uint64_t size);
ACL_FUNC_VISIBILITY aclFloatArray* aclCreateFloatArray(const float* value, uint64_t size);
ACL_FUNC_VISIBILITY aclBoolArray* aclCreateBoolArray(const bool* value, uint64_t size);
ACL_FUNC_VISIBILITY aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size);
ACL_FUNC_VISIBILITY aclScalarList* aclCreateScalarList(const aclScalar* const* value, uint64_t size);

ACL_FUNC_VISIBILITY aclnnStatus aclDestroyTensor(const aclTensor* tensor);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyScalar(const aclScalar* scalar);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyIntArray(const aclIntArray* array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyFloatArray(const aclFloatArray* array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyBoolArray(const aclBoolArray* array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyTensorList(const aclTensorList* array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyScalarList(const aclScalarList* array);
ACL_FUNC_VISIBILITY aclnnStatus aclGetViewShape(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
ACL_FUNC_VISIBILITY aclnnStatus aclGetStorageShape(const aclTensor* tensor, int64_t** storageDims,
                                                   uint64_t* storageDimsNum);
ACL_FUNC_VISIBILITY aclnnStatus aclGetViewStrides(const aclTensor* tensor, int64_t** stridesValue,
                                                  uint64_t* stridesNum);
ACL_FUNC_VISIBILITY aclnnStatus aclGetViewOffset(const aclTensor* tensor, int64_t* offset);
ACL_FUNC_VISIBILITY aclnnStatus aclGetFormat(const aclTensor* tensor, aclFormat* format);
ACL_FUNC_VISIBILITY aclnnStatus aclGetDataType(const aclTensor* tensor, aclDataType* dataType);
ACL_FUNC_VISIBILITY aclnnStatus aclGetIntArraySize(const aclIntArray* array, uint64_t* size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetFloatArraySize(const aclFloatArray* array, uint64_t* size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetBoolArraySize(const aclBoolArray* array, uint64_t* size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetTensorListSize(const aclTensorList* tensorList, uint64_t* size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetScalarListSize(const aclScalarList* scalarList, uint64_t* size);

ACL_FUNC_VISIBILITY aclnnStatus aclInitTensor(aclTensor* tensor, const int64_t* viewDims, uint64_t viewDimsNum,
                                              aclDataType dataType, const int64_t* stride, int64_t offset,
                                              aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
                                              void* tensorDataAddr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor);
ACL_FUNC_VISIBILITY aclnnStatus AclSetInputTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor,
                                                      void* addr);
ACL_FUNC_VISIBILITY aclnnStatus AclSetOutputTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor,
                                                       void* addr);
ACL_FUNC_VISIBILITY aclnnStatus AclSetDynamicInputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                                             const size_t relativeIndex, aclTensorList* tensors,
                                                             void* addr);
ACL_FUNC_VISIBILITY aclnnStatus AclSetDynamicOutputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                                              const size_t relativeIndex, aclTensorList* tensors,
                                                              void* addr);
ACL_FUNC_VISIBILITY aclnnStatus AclSetTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor,
                                                 void* addr);
ACL_FUNC_VISIBILITY aclnnStatus AclSetDynamicTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                                        const size_t relativeIndex, aclTensorList* tensors, void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetInputTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor,
                                                      void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetOutputTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor,
                                                       void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetDynamicInputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                                             const size_t relativeIndex, aclTensorList* tensors,
                                                             void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetDynamicOutputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                                              const size_t relativeIndex, aclTensorList* tensors,
                                                              void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor,
                                                 void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetDynamicTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                                        const size_t relativeIndex, aclTensorList* tensors, void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclSetRawTensorAddr(aclTensor* tensor, void* addr);
ACL_FUNC_VISIBILITY aclnnStatus aclGetRawTensorAddr(const aclTensor* tensor, void** addr);
/**
 * @ingroup AscendCL Dump
 * @brief Custom operator dump the input/output data for accuracy comparison
 * @attention Opened for custom operator, not recommended for internal(official compatibility time: 2027/01/30)
 * @param [in] opType: Operator type
 * @param [in] opName: Operator name
 * @param [in] tensors: aclTensor array(input tensor first, output tensor last)
 * @param [in] inputTensorNum: Input tensor number in aclTensor array
 * @param [in] outputTensorNum: Output tensor number in aclTensor array
 * @param [in] stream: Stream pointer
 * @retval 0: success, other value: failure
 * @since Created on 2026/01/30
 */
ACL_FUNC_VISIBILITY aclnnStatus aclDumpOpTensors(const char* opType, const char* opName, aclTensor** tensors,
                                                 size_t inputTensorNum, size_t outputTensorNum, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_OP_API_COMMON_INC_EXTERNAL_ACL_META_H
