/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NNOPBASE_API_H_
#define NNOPBASE_API_H_

#include "aclnn/acl_meta.h"
#include "graph/types.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/continuous_vector.h"

void NnopbaseOpLogE(const aclnnStatus code, const char* const expr);

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define VISIBILITY_EXPORT _declspec(dllexport)
#else
#define VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    uint32_t id;
    const char* funcName;
    bool hasReg;
} NnopbaseDfxId;

typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;

typedef struct {
    TensorDesc* inputsDesc;
    size_t inputsNum;
    TensorDesc* outputsDesc;
    size_t outputsNum;
} SupportInfo;

typedef struct {
    SupportInfo* supportInfo;
    size_t num;
} OpSocSupportInfo;

typedef struct {
    OpSocSupportInfo* socSupportInfo;
    size_t num;
} OpSupportList;

enum NnopbaseAttrDtype { kNnopbaseBool = 0U, kNnopbaseFloat, kNnopbaseInt, kNnopbaseString, kNnopbaseAttrEnd };

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

VISIBILITY_EXPORT aclnnStatus NnopbaseCreateExecutorSpace(void** space);
VISIBILITY_EXPORT void* NnopbaseGetExecutor(void* space, const char* opType, char* inputsDesc, uint32_t inputNum,
                                            char* outputsDesc, uint32_t outputNum, char* attrsDesc, uint32_t attrsNum);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddInput(void* executor, const aclTensor* tensor, const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddIntArrayInput(void* executor, const aclIntArray* array, const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddBoolArrayInput(void* executor, const aclBoolArray* array,
                                                        const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddFloatArrayInput(void* executor, const aclFloatArray* array,
                                                         const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddIgnoreContinuesInput(void* executor, const aclTensor* tensor,
                                                              const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetAttrAddr(void* executor, const size_t index, void** attrAddr, size_t* attrLen);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddOutput(void* executor, const aclTensor* tensor, const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddDynamicInput(void* executor, const aclTensorList* tensorList,
                                                      const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddIgnoreContiguousDynamicInput(void* executor, const aclTensorList* tensorList,
                                                                      const uint32_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddDynamicOutput(void* executor, const aclTensorList* tensorList,
                                                       const uint32_t index);

// 属性添加接口
VISIBILITY_EXPORT aclnnStatus NnopbaseAddAttr(void* executor, void* attrAddr, const size_t attrLen, const size_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddAttrWithDtype(void* executor, void* attrAddr, const size_t attrLen,
                                                       const size_t index, const NnopbaseAttrDtype dtype);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddIntArrayAttr(void* executor, const aclIntArray* array, const size_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddBoolArrayAttr(void* executor, const aclBoolArray* array, const size_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddFloatArrayAttr(void* executor, const aclFloatArray* array, const size_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddArrayAttr(void* executor, void* array, const size_t len,
                                                   const size_t elementSize, const size_t index);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddArrayAttrWithDtype(void* executor, void* array, const size_t len,
                                                            const size_t elementSize, const size_t index,
                                                            const NnopbaseAttrDtype dtype);

// profiling上报接口
VISIBILITY_EXPORT uint64_t NnopbaseMsprofSysTime();
VISIBILITY_EXPORT aclnnStatus NnopbaseAddTilingId(void* executor, NnopbaseDfxId* tilingId);
VISIBILITY_EXPORT void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
VISIBILITY_EXPORT void NnopbaseReportLaunchInfo(const uint64_t beginTime, const char* const opType);
VISIBILITY_EXPORT aclnnStatus NnopbaseReportAicpuAdditionInfo(const uint64_t timeStamp, const char* const opType);

VISIBILITY_EXPORT aclnnStatus NnopbaseRunForWorkspace(void* executor, uint64_t* workspaceLen);
VISIBILITY_EXPORT aclnnStatus NnopbaseRunWithWorkspace(void* executor, aclrtStream stream, void* workspace,
                                                       uint64_t workspaceLen);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddSupportList(void* executor, OpSupportList* list, uint32_t* socSupportList,
                                                     size_t socSupportListLen);
VISIBILITY_EXPORT void NnopbaseAddOpTypeId(void* executor, const uint32_t opTypeId);

// 获取非连续相关接口
VISIBILITY_EXPORT aclnnStatus NnopbaseGetUnContiguousTensors(void* executor, const aclTensorList** inTensors);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetUnContExecutor(void* executor, aclOpExecutor* inExe, const size_t inWsSize);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetUnContExecutor(void* executor, aclOpExecutor** inExe, size_t* inWsSize);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetRefUnContiguousTensors(void* executor, const aclTensorList** unContTensors,
                                                                const aclTensorList** contTensors);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetRef(void* executor, const size_t inputIrIdx, const size_t outputIrIdx);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetViewCopyExecutor(void* executor, aclOpExecutor* exe);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetViewCopyExecutor(void* executor, aclOpExecutor** exe);
VISIBILITY_EXPORT aclnnStatus NnopbaseReleaseRefContiguousTensors(void* executor, const aclTensorList** tensors);

VISIBILITY_EXPORT void NnopbaseGetTilingData(void* executor, void** tilingData, uint64_t* dataLen);
VISIBILITY_EXPORT void NnopbaseGetInputTensorAddr(void* executor, size_t index, void** addr);
VISIBILITY_EXPORT void NnopbaseGetOutputTensorAddr(void* executor, size_t index, void** addr);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetInputTensorCount(const void* const executor, const size_t irIndex,
                                                          uint32_t* const count);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetOutputTensorCount(const void* const executor, const size_t irIndex,
                                                           uint32_t* const count);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetDynamicInputTensorAddrs(const void* const executor, const size_t irIndex,
                                                                 void** const addrs, const size_t addrSize,
                                                                 uint32_t* const count);
VISIBILITY_EXPORT aclnnStatus NnopbaseGetDynamicOutputTensorAddrs(const void* const executor, const size_t irIndex,
                                                                  void** const addrs, const size_t addrSize,
                                                                  uint32_t* const count);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetInputTensorAddr(void* executor, const size_t index, const void* const addr);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetOutputTensorAddr(void* executor, const size_t index, const void* const addr);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetDynamicInputTensorAddr(void* executor, size_t irIndex,
                                                                const size_t relativeIndex, const void* const addr);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetDynamicOutputTensorAddr(void* executor, size_t irIndex,
                                                                 const size_t relativeIndex, const void* const addr);
VISIBILITY_EXPORT void NnopbaseSetUserHandle(void* executor, void* handle);
VISIBILITY_EXPORT void* NnopbaseGetUserHandle(void* executor);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddScalarInput(void* executor, const aclScalar* scalar, const uint32_t index,
                                                     const int32_t srcIndex, const ge::DataType dtype);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddScalarListInput(void* executor, const aclScalarList* scalarList,
                                                         const uint32_t index, const int32_t srcIndex,
                                                         const ge::DataType dtype);
VISIBILITY_EXPORT aclnnStatus NnopbaseAddParamName(void* executor, const uint32_t index, const char* name,
                                                   const bool isInput);

VISIBILITY_EXPORT void NnopbaseSetMatchArgsFlag(void* executor);
VISIBILITY_EXPORT bool NnopbaseMatchArgs(void* executor, uint64_t* workspaceLen);
VISIBILITY_EXPORT const char* NnopbaseGetSocName();
VISIBILITY_EXPORT aclnnStatus NnopbaseAddSocNameList(void* executor, OpSupportList* list,
                                                     const char* const* socNameList, size_t socNameListLen);
VISIBILITY_EXPORT bool NnopbaseSupportTensorV2();

// for mc2 op
VISIBILITY_EXPORT aclnnStatus NnopbaseSetMc2(void* const executor);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetHcomGroup(void* const executor, char* const group);
VISIBILITY_EXPORT void NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);
VISIBILITY_EXPORT void NnopbaseSetHcclServerTypeList(void* executor, NnopbaseHcclServerType* hcclServerTypeList,
                                                     const uint32_t* socSupportList, size_t socSupportListLen);
VISIBILITY_EXPORT aclnnStatus NnopbaseSetHcclServerTypeBySocName(void* executor,
                                                                 NnopbaseHcclServerType* hcclServerTypeList,
                                                                 const char* const* socNameList, size_t socNameListLen);

VISIBILITY_EXPORT void NnopbaseSetZeroEleOutputLaunchFlag(void* executor);
VISIBILITY_EXPORT void* NnopbaseGetApiFunc(const char* funcName);
VISIBILITY_EXPORT
aclnnStatus NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);
VISIBILITY_EXPORT
aclnnStatus NnopbaseAddOutputShapeDependTensor(void* executor, aclTensor* tensor, const uint32_t index);

VISIBILITY_EXPORT aclnnStatus NnopbaseSetFormatMatchMode(void* executor, const uint32_t mode);
VISIBILITY_EXPORT void NnopbaseSetParamCheckMode(void* executor, const uint32_t mode);

// for fallback
VISIBILITY_EXPORT void* NnopbaseGetOpApiFunc(const char* funcName);
VISIBILITY_EXPORT aclTensor* NnopbaseConvertTensor(const gert::Tensor* tensor);
VISIBILITY_EXPORT aclTensorList* NnopbaseConvertTensorList(std::vector<const gert::Tensor*>& tensorList);
VISIBILITY_EXPORT aclBoolArray* NnopbaseCovertBoolArray(const gert::Tensor* tensor);
VISIBILITY_EXPORT aclIntArray* NnopbaseCovertIntArray(const gert::Tensor* tensor);
VISIBILITY_EXPORT aclFloatArray* NnopbaseCovertFloatArray(const gert::Tensor* tensor);
VISIBILITY_EXPORT aclScalar* NnopbaseConvertScalar(const gert::Tensor* tensor);
VISIBILITY_EXPORT aclScalarList* NnopbaseConvertScalarList(const gert::Tensor* tensor);
VISIBILITY_EXPORT aclIntArray* NnopbaseCovertIntArrayAttr(const gert::TypedContinuousVector<int64_t>* arr);
VISIBILITY_EXPORT aclBoolArray* NnopbaseCovertBoolArrayAttr(const gert::TypedContinuousVector<bool>* arr);
VISIBILITY_EXPORT aclFloatArray* NnopbaseCovertFloatArrayAttr(const gert::TypedContinuousVector<float>* arr);
VISIBILITY_EXPORT void NnopbaseDestroyTensor(const aclTensor* tensor);
VISIBILITY_EXPORT void NnopbaseDestroyTensorList(const aclTensorList* tensorList);
VISIBILITY_EXPORT void NnopbaseDestroyScalar(const aclScalar* scalar);
VISIBILITY_EXPORT void NnopbaseDestroyScalarList(const aclScalarList* scalar);
VISIBILITY_EXPORT void NnopbaseDestroyIntArray(const aclIntArray* array);
VISIBILITY_EXPORT void NnopbaseDestroyBoolArray(const aclBoolArray* array);
VISIBILITY_EXPORT void NnopbaseDestroyFloatArray(const aclFloatArray* array);
#ifdef __cplusplus
}
#endif
#endif
