/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include "aclnn/aclnn_base.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"
#include "individual_op_api.h"
#include "executor/indv_executor.h"
#include "kernel_utils.h"
#include "op_cache_internal.h"
#include "kernel_mgr.h"
#include "dlopen_api.h"
#include "op_dfx_internal.h"
#include "file_utils.h"
#include "bridge_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

aclTensor* aclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride,
                           int64_t offset, aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
                           void* tensorData)
{
    if ((viewDims == nullptr && viewDimsNum > 0) || (storageDims == nullptr && storageDimsNum > 0)) {
        return nullptr;
    }
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(viewDims, viewDimsNum, dataType, stride, offset, format, storageDims,
                                         storageDimsNum, tensorData);
                  return tensor;, OP_LOGE(ACLNN_ERR_INNER, "aclCreateTensor error."); delete tensor; return nullptr;);
}

aclScalar* aclCreateScalar(void* value, aclDataType dataType)
{
    if (value == nullptr) {
        return nullptr;
    }
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value, op::ToOpDataType(dataType)); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclCreateScalar error."); delete scalar; return nullptr;);
}

aclIntArray* aclCreateIntArray(const int64_t* value, uint64_t size)
{
    aclIntArray* intArray = nullptr;
    ADD_TRY_CATCH(intArray = new aclIntArray(value, size); return intArray;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclCreateIntArray error."); delete intArray; return nullptr;);
}

aclFloatArray* aclCreateFloatArray(const float* value, uint64_t size)
{
    aclFloatArray* floatArray = nullptr;
    ADD_TRY_CATCH(floatArray = new aclFloatArray(value, size); return floatArray;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclCreateFloatArray error."); delete floatArray; return nullptr;);
}

aclBoolArray* aclCreateBoolArray(const bool* value, uint64_t size)
{
    aclBoolArray* boolArray = nullptr;
    ADD_TRY_CATCH(boolArray = new aclBoolArray(value, size); return boolArray;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclCreateBoolArray error."); delete boolArray; return nullptr;);
}

aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size)
{
    aclTensorList* tensorList = nullptr;
    ADD_TRY_CATCH(tensorList = new aclTensorList(value, size); return tensorList;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclCreateTensorList error."); delete tensorList; return nullptr;);
}

aclScalarList* aclCreateScalarList(const aclScalar* const* value, uint64_t size)
{
    aclScalarList* scalarList = nullptr;
    ADD_TRY_CATCH(scalarList = new aclScalarList(value, size); return scalarList;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclCreateScalarList error."); delete scalarList; return nullptr;);
}

aclnnStatus aclDestroyTensor(const aclTensor* tensor)
{
    if (tensor == nullptr) {
        return OK;
    }
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclTensor*>(tensor))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(tensor));
    }
    delete tensor;
    return OK;
}

aclnnStatus aclDestroyScalar(const aclScalar* scalar)
{
    if (scalar == nullptr) {
        return OK;
    }
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclScalar*>(scalar))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(scalar));
    }
    delete scalar;
    return OK;
}

aclnnStatus aclDestroyIntArray(const aclIntArray* array)
{
    if (array == nullptr) {
        return OK;
    }
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclIntArray*>(array))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(array));
    }
    delete array;
    return OK;
}

aclnnStatus aclDestroyFloatArray(const aclFloatArray* array)
{
    if (array == nullptr) {
        return OK;
    }
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclFloatArray*>(array))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(array));
    }
    delete array;
    return OK;
}

aclnnStatus aclDestroyBoolArray(const aclBoolArray* array)
{
    if (array == nullptr) {
        return OK;
    }
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclBoolArray*>(array))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(array));
    }
    delete array;
    return OK;
}

aclnnStatus aclDestroyTensorList(const aclTensorList* array)
{
    if (array == nullptr) {
        return OK;
    }
    // 仅对外层 list 指针本身做一次预检；元素由循环内 aclDestroyTensor 各自覆盖
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclTensorList*>(array))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(array));
    }
    for (uint64_t i = 0; i < array->Size(); i++) {
        aclDestroyTensor((*array)[i]);
    }
    delete array;
    return OK;
}

aclnnStatus aclDestroyScalarList(const aclScalarList* array)
{
    if (array == nullptr) {
        return OK;
    }
    // 仅对外层 list 指针本身做一次预检；元素由循环内 aclDestroyScalar 各自覆盖
    if (unlikely(op::internal::IsAclnnDebugEnabled()) &&
        op::internal::CheckDoubleFree(const_cast<aclScalarList*>(array))) {
        OP_LOGW("possible double-free at addr %p.", static_cast<const void*>(array));
    }
    for (uint64_t i = 0; i < array->Size(); i++) {
        aclDestroyScalar((*array)[i]);
    }
    delete array;
    return OK;
}

aclnnStatus aclGetViewShape(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum)
{
    if (tensor == nullptr || viewDims == nullptr || viewDimsNum == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto dim_num = tensor->GetViewShape().GetDimNum();
    *viewDimsNum = dim_num;
    *viewDims = new (std::nothrow) int64_t[dim_num];
    CHECK_RET(*viewDims != nullptr, ACLNN_ERR_INNER_NULLPTR);

    for (size_t i = 0; i < dim_num; i++) {
        (*viewDims)[i] = tensor->GetViewShape().GetDim(i);
    }

    return OK;
}

aclnnStatus aclGetViewStrides(const aclTensor* tensor, int64_t** stridesValue, uint64_t* stridesNum)
{
    if (tensor == nullptr || stridesValue == nullptr || stridesNum == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto stridesCount = tensor->GetViewStrides().size();
    *stridesNum = stridesCount;
    *stridesValue = new (std::nothrow) int64_t[stridesCount];
    CHECK_RET(*stridesValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
    for (size_t i = 0; i < stridesCount; i++) {
        (*stridesValue)[i] = tensor->GetViewStrides()[i];
    }
    return OK;
}

aclnnStatus aclGetStorageShape(const aclTensor* tensor, int64_t** storageDims, uint64_t* storageDimsNum)
{
    if (tensor == nullptr || storageDims == nullptr || storageDimsNum == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto storageCount = tensor->GetStorageShape().GetDimNum();
    *storageDimsNum = storageCount;
    *storageDims = new (std::nothrow) int64_t[storageCount];
    CHECK_RET(*storageDims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    for (size_t i = 0; i < storageCount; i++) {
        (*storageDims)[i] = tensor->GetStorageShape().GetDim(i);
    }
    return OK;
}

aclnnStatus aclGetViewOffset(const aclTensor* tensor, int64_t* offset)
{
    if (tensor == nullptr || offset == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *offset = tensor->GetViewOffset();
    return OK;
}

aclnnStatus aclGetFormat(const aclTensor* tensor, aclFormat* format)
{
    if (tensor == nullptr || format == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *format = op::ToAclFormat(tensor->GetViewFormat());
    return OK;
}

aclnnStatus aclGetDataType(const aclTensor* tensor, aclDataType* dataType)
{
    if (tensor == nullptr || dataType == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *dataType = op::ToAclDataType(tensor->GetDataType());
    return OK;
}

aclnnStatus aclGetIntArraySize(const aclIntArray* array, uint64_t* size)
{
    if (array == nullptr || size == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *size = array->Size();
    return OK;
}

aclnnStatus aclGetFloatArraySize(const aclFloatArray* array, uint64_t* size)
{
    if (array == nullptr || size == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *size = array->Size();
    return OK;
}

aclnnStatus aclGetBoolArraySize(const aclBoolArray* array, uint64_t* size)
{
    if (array == nullptr || size == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *size = array->Size();
    return OK;
}

aclnnStatus aclGetTensorListSize(const aclTensorList* tensorList, uint64_t* size)
{
    if (tensorList == nullptr || size == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *size = tensorList->Size();
    return OK;
}

aclnnStatus aclGetScalarListSize(const aclScalarList* scalarList, uint64_t* size)
{
    if (scalarList == nullptr || size == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *size = scalarList->Size();
    return OK;
}

aclnnStatus aclInitTensor(aclTensor* tensor, const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                          const int64_t* stride, int64_t offset, aclFormat format, const int64_t* storageDims,
                          uint64_t storageDimsNum, void* tensorDataAddr)
{
    if (tensor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    tensor->InitTensor(viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum,
                       tensorDataAddr);
    return OK;
}

aclnnStatus aclSetInputTensorAddr(aclOpExecutor* executor, [[maybe_unused]] const size_t index, aclTensor* tensor,
                                  void* addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensor, ACLNN_ERR_INNER_NULLPTR);
    tensor->SetStorageAddr(addr);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        size_t realIndex = index;
        NnopbaseGetRealIndex((op::internal::PtrCastTo<NnopbaseExecutor>(executor))->args->inputs.paramDescs,
                             &realIndex);
        return NnopbaseSetInputTensorAddr(executor, realIndex, tensor->GetData());
    }
    return OK;
}

aclnnStatus AclSetInputTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor, void* addr)
{
    return aclSetInputTensorAddr(executor, index, tensor, addr);
}

aclnnStatus aclSetOutputTensorAddr(aclOpExecutor* executor, [[maybe_unused]] const size_t index, aclTensor* tensor,
                                   void* addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensor, ACLNN_ERR_INNER_NULLPTR);
    tensor->SetStorageAddr(addr);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        return NnopbaseSetOutputTensorAddr(executor, index, tensor->GetData());
    }
    return OK;
}

aclnnStatus AclSetOutputTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor, void* addr)
{
    return aclSetOutputTensorAddr(executor, index, tensor, addr);
}

aclnnStatus aclSetDynamicInputTensorAddr(aclOpExecutor* executor, size_t irIndex, const size_t relativeIndex,
                                         aclTensorList* tensors, void* addr)
{
    OP_LOGI("executor addr is %p, irIndex is %zu, relativeIndex is %zu.", executor, irIndex, relativeIndex);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensors, ACLNN_ERR_INNER_NULLPTR);
    CHECK_COND((relativeIndex < tensors->Size()), ACLNN_ERR_PARAM_INVALID,
               "Set dynamic input tensor addr failed. "
               "relativeIndex[%zu] is out of tensors size[%lu].",
               relativeIndex, tensors->Size());

    auto tensor = (*tensors)[relativeIndex];
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    tensor->SetStorageAddr(addr);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        size_t tensorIrIndex = irIndex;
        NnopbaseGetRealIndex((op::internal::PtrCastTo<NnopbaseExecutor>(executor))->args->inputs.paramDescs,
                             &tensorIrIndex);
        return NnopbaseSetDynamicInputTensorAddr(executor, tensorIrIndex, relativeIndex, tensor->GetData());
    }
    return OK;
}

aclnnStatus AclSetDynamicInputTensorAddr(aclOpExecutor* executor, size_t irIndex, const size_t relativeIndex,
                                         aclTensorList* tensors, void* addr)
{
    return aclSetDynamicInputTensorAddr(executor, irIndex, relativeIndex, tensors, addr);
}

aclnnStatus aclSetDynamicOutputTensorAddr(aclOpExecutor* executor, size_t irIndex, const size_t relativeIndex,
                                          aclTensorList* tensors, void* addr)
{
    OP_LOGI("executor addr is %p, irIndex is %zu, relativeIndex is %zu.", executor, irIndex, relativeIndex);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensors, ACLNN_ERR_INNER_NULLPTR);
    CHECK_COND((relativeIndex < tensors->Size()), ACLNN_ERR_PARAM_INVALID,
               "Set dynamic output tensor addr failed. "
               "relativeIndex[%zu] is out of tensors size[%lu].",
               relativeIndex, tensors->Size());
    auto tensor = (*tensors)[relativeIndex];

    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    tensor->SetStorageAddr(addr);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        return NnopbaseSetDynamicOutputTensorAddr(executor, irIndex, relativeIndex, tensor->GetData());
    }
    return OK;
}

aclnnStatus AclSetDynamicOutputTensorAddr(aclOpExecutor* executor, size_t irIndex, const size_t relativeIndex,
                                          aclTensorList* tensors, void* addr)
{
    return aclSetDynamicOutputTensorAddr(executor, irIndex, relativeIndex, tensors, addr);
}

aclnnStatus aclSetTensorAddr(aclOpExecutor* executor, [[maybe_unused]] const size_t index, aclTensor* tensor,
                             void* addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensor, ACLNN_ERR_INNER_NULLPTR);
    tensor->SetStorageAddr(addr);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);

    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        size_t tensorIndex = 0U;
        if (NnopbaseIsInput(executor, index, &tensorIndex)) {
            return NnopbaseSetInputTensorAddr(executor, tensorIndex, tensor->GetData());
        } else {
            return NnopbaseSetOutputTensorAddr(executor, tensorIndex, tensor->GetData());
        }
    }
    return OK;
}

aclnnStatus AclSetTensorAddr(aclOpExecutor* executor, const size_t index, aclTensor* tensor, void* addr)
{
    return aclSetTensorAddr(executor, index, tensor, addr);
}

aclnnStatus aclSetDynamicTensorAddr(aclOpExecutor* executor, size_t irIndex, const size_t relativeIndex,
                                    aclTensorList* tensors, void* addr)
{
    OP_LOGI("executor addr is %p, irIndex is %zu, relativeIndex is %zu.", executor, irIndex, relativeIndex);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensors, ACLNN_ERR_INNER_NULLPTR);
    CHECK_COND((relativeIndex < tensors->Size()), ACLNN_ERR_PARAM_INVALID,
               "Set dynamic tensor addr failed. "
               "relativeIndex[%zu] is out of tensors size[%lu].",
               relativeIndex, tensors->Size());

    auto tensor = (*tensors)[relativeIndex];
    tensor->SetStorageAddr(addr);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        size_t tensorIrIndex = 0U;
        if (NnopbaseDynamicIsInput(executor, irIndex, &tensorIrIndex)) {
            return NnopbaseSetDynamicInputTensorAddr(executor, tensorIrIndex, relativeIndex, tensor->GetData());
        } else {
            return NnopbaseSetDynamicOutputTensorAddr(executor, tensorIrIndex, relativeIndex, tensor->GetData());
        }
    }
    return OK;
}

aclnnStatus AclSetDynamicTensorAddr(aclOpExecutor* executor, size_t irIndex, const size_t relativeIndex,
                                    aclTensorList* tensors, void* addr)
{
    return aclSetDynamicTensorAddr(executor, irIndex, relativeIndex, tensors, addr);
}

aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    OP_LOGI("executor addr is %p, magicNum is 0x%lX, executor->GetMagicNumber() is 0x%lX.", executor, *magicNum,
            executor->GetMagicNumber());
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        return NnopbaseSetRepeatable(executor);
    } else if (executor->GetMagicNumber() == K_EXECUTOR_MAGIC_NUMBER) {
        return executor->SetRepeatable();
    }
    return ACLNN_ERR_INNER;
}

aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_INNER_NULLPTR);
    uint64_t* magicNum = op::internal::PtrCastTo<uint64_t>(executor);
    OP_LOGI("executor addr is %p, magicNum is 0x%lX, executor->GetMagicNumber() is 0x%lX.", executor, *magicNum,
            executor->GetMagicNumber());
    if (*magicNum == NNOPBASE_EXECUTOR_MAGIC_NUMBER) {
        return NnopbaseResetExecutor(executor);
    } else if (executor->GetMagicNumber() == K_EXECUTOR_MAGIC_NUMBER) {
        delete executor;
        return OK;
    } else if (*magicNum == op::internal::K_CACHE_WRAP_MAGIC_NUMBER) {
        op::internal::OpExecCacheWrap* cache = op::internal::PtrCastTo<op::internal::OpExecCacheWrap>(executor);
        delete cache;
        return OK;
    }
    return ACLNN_ERR_INNER;
}

aclnnStatus aclSetRawTensorAddr(aclTensor* tensor, void* addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensor, ACLNN_ERR_PARAM_NULLPTR);
    tensor->SetStorageAddr(addr);
    return OK;
}

aclnnStatus aclGetRawTensorAddr(const aclTensor* tensor, void** addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensor, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(addr, ACLNN_ERR_PARAM_NULLPTR);
    *addr = tensor->GetStorageAddr();
    return OK;
}

aclnnStatus aclnnReselectStaticKernel()
{
    OP_LOGI("Start reselect static kernel.");
    op::internal::ReinitOpCacheManager();
    op::internal::gKernelMgr.ReloadStaticBinJson();
    NnopbaseReloadStaticBinJsonInfos(nullptr);
    return OK;
}

// realPath.c_str() 为局部指针，下游 ReloadStaticBinJson / NnopbaseReloadStaticBinJsonInfos
// 必须同步消费，不得异步持有
aclnnStatus aclnnReselectStaticKernelWithPath(const char* staticKernelPath)
{
    if (staticKernelPath == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input staticKernelPath is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string realPath = op::RealPath(staticKernelPath);
    if (realPath.empty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid static kernel path: %s.", staticKernelPath);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("Start reselect static kernel, input path: %s, real path: %s", staticKernelPath, realPath.c_str());
    op::internal::ReinitOpCacheManager();
    op::internal::gKernelMgr.ReloadStaticBinJson(realPath.c_str());
    NnopbaseReloadStaticBinJsonInfos(realPath.c_str());
    return OK;
}

aclnnStatus aclDumpOpTensors(const char* opType, const char* opName, aclTensor** tensors, size_t inputTensorNum,
                             size_t outputTensorNum, aclrtStream stream)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(opType, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(opName, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(tensors, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(stream, ACLNN_ERR_PARAM_NULLPTR);
    OP_LOGI("start dump tensors. opType: %s, opName: %s.", opType, opName);
    std::vector<Adx::TensorInfoV2> dumpTensors;
    std::vector<const aclTensor*> inputTensors;
    for (size_t i = 0U; i < inputTensorNum; i++) {
        inputTensors.push_back(tensors[i]);
    }
    op::internal::PrepareL2DumpTensor(dumpTensors, inputTensors, op::OpIOType::OpInputType);
    std::vector<const aclTensor*> outputTensors;
    for (size_t i = inputTensorNum; i < inputTensorNum + outputTensorNum; i++) {
        outputTensors.push_back(tensors[i]);
    }
    op::internal::PrepareL2DumpTensor(dumpTensors, outputTensors, op::OpIOType::OpOutputType);
    return Adx::AdumpDumpTensorV2(opType, opName, dumpTensors, stream);
}

#ifdef __cplusplus
}
#endif
