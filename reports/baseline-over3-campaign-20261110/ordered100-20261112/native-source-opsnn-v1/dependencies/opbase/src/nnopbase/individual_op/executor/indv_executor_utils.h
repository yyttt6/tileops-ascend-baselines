/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_EXECUTOR_UTILS_H_
#define INDV_EXECUTOR_UTILS_H_

#include <vector>
#include <unordered_map>
#include "utils/indv_base.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "indv_bininfo.h"
#include "mmpa/mmpa_api.h"

namespace nnopbase {
constexpr uint32_t NNOPBASE_ENV_VAR_LENGH = 128U;
constexpr const char* IMPL_MODE_HIGH_PERFORMANCE = "high_performance";
constexpr const char* IMPL_MODE_HIGH_PRECISION = "high_precision";

template <typename T>
NnopbaseUChar* NnopbaseAppendByte(NnopbaseUChar* buf, T src)
{
    T* dst = op::internal::PtrCastTo<T>(buf);
    *dst = src;
    return buf + sizeof(T);
}

inline bool EnableNnopbaseTimeStamp()
{
    const char* profilingToStdOut = nullptr;
    MM_SYS_GET_ENV(MM_ENV_GE_PROFILING_TO_STD_OUT, profilingToStdOut);
    return ((profilingToStdOut != nullptr) && strcmp(profilingToStdOut, "1") == 0);
}

inline bool EnableNnopbaseArgsCache()
{
    const char* disAbleArgsCache = nullptr;
    MM_SYS_GET_ENV(MM_ENV_DISABLE_L2_CACHE, disAbleArgsCache);
    return disAbleArgsCache == nullptr ? true : strcmp(disAbleArgsCache, "1") != 0; // 1表示开启
}

inline int64_t GetGlobalDeterministic()
{
    int64_t deterministicLevel = 0;                                                       // 0表示非确定性
    const aclError ret = aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicLevel); // SYS_OPT_DETERMINISTIC
    OP_CHECK_NO_RETURN(ret == ACL_SUCCESS, deterministicLevel = 0;
                       OP_LOGW("Failed to get deterministic from system, ret= %d.", ret));
    if (deterministicLevel == 1) {
        int64_t consistency = 0;
        aclError consistencyRet = aclrtGetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, &consistency);
        OP_CHECK_NO_RETURN(consistencyRet == ACL_SUCCESS, consistency = 0;
                           OP_LOGW("Failed to get consistency from system, ret= %d.", consistencyRet));
        if (consistency == 1) {
            deterministicLevel = 2; // 2表示强一致性
        }
    }
    OP_LOGI("determinisic level of system is %lld", deterministicLevel);
    return deterministicLevel;
}

inline void NnopbaseGetCoreNum(uint32_t* const aicNum, uint32_t* const aivNum)
{
    uint32_t coreNum = 0;
    auto ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &coreNum);
    OP_LOGD("Get cube core num %d, ret %d.", coreNum, ret);
    *aicNum = coreNum;

    ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &coreNum);
    OP_LOGD("Get vector core num %d, ret %d.", coreNum, ret);
    *aivNum = coreNum;
}

inline std::string NnopbaseGetImplMode(const NnopbaseUChar precision)
{
    if (precision == 1) {
        return IMPL_MODE_HIGH_PERFORMANCE;
    } else if (precision == 2) { // 2 is precision
        return IMPL_MODE_HIGH_PRECISION;
    } else {
        return "";
    }
}

inline aclnnStatus NnopbaseSetOverFlowAddr(void*& addr)
{
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_SATURATION;
    const aclError ret = aclrtGetDeviceSatMode(&floatOverflowMode);
    OP_LOGD("Get DeviceSatMode ret = %d.", ret);
    if ((floatOverflowMode == ACL_RT_OVERFLOW_MODE_SATURATION) || (floatOverflowMode == ACL_RT_OVERFLOW_MODE_INFNAN)) {
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtCtxGetFloatOverflowAddr(&addr));
        NNOPBASE_ASSERT_NOTNULL_RETVAL(addr);
    } else {
        addr = nullptr;
    }
    return OK;
}

inline bool GetDebugKernel()
{
    int64_t value = 0;                                                                // 0表示不开启debug kernel
    const aclError ret = aclrtCtxGetSysParamOpt(ACL_OPT_ENABLE_DEBUG_KERNEL, &value); // SYS_OPT_ENABLE_DEBUG_KERNEL = 1
    OP_LOGD("Get system param debug kernel ret = %d.", ret);
    return (value == 1);                                                              // 1表示开启debug kernel
}

static inline void NnopbaseSetDtypeAndSize(const aclIntArray* array, GertTensor* rt2Tensor)
{
    rt2Tensor->SetDataType(ge::DataType::DT_INT64);
    rt2Tensor->SetSize(static_cast<size_t>(array->Size()) * sizeof(int64_t));
}

static inline void NnopbaseSetDtypeAndSize(const aclBoolArray* array, GertTensor* rt2Tensor)
{
    rt2Tensor->SetDataType(ge::DataType::DT_BOOL);
    rt2Tensor->SetSize(static_cast<size_t>(array->Size()) * sizeof(bool));
}

static inline void NnopbaseSetDtypeAndSize(const aclFloatArray* array, GertTensor* rt2Tensor)
{
    rt2Tensor->SetDataType(ge::DataType::DT_FLOAT);
    rt2Tensor->SetSize(static_cast<size_t>(array->Size()) * sizeof(float));
}

template <typename T>
aclnnStatus NnopbaseSaveArray(const T* array, NnopbaseTensor* tensor)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(array);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tensor);
    tensor->isNull = false;

    GertTensor* rt2Tensor = &tensor->rt2Tensor;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rt2Tensor);
    rt2Tensor->MutableOriginShape() = {static_cast<int64_t>(array->Size())};
    rt2Tensor->MutableStorageShape() = {static_cast<int64_t>(array->Size())};
    rt2Tensor->SetOriginFormat(ge::FORMAT_ND);
    rt2Tensor->SetStorageFormat(ge::FORMAT_ND);
    rt2Tensor->MutableTensorData().SetPlacement(gert::kOnHost); // valuedepend一定是host内存
    // valueDepend 需要获取 tensorSize
    NnopbaseSetDtypeAndSize(array, rt2Tensor);

    NNOPBASE_ASSERT_TRUE_RETVAL((rt2Tensor->MutableTensorData().SetAddr(array->GetData(), nullptr)) ==
                                ge::GRAPH_SUCCESS);
    rt2Tensor->MutableStride().SetDimNum(0U);
    rt2Tensor->SetOffset(0);
    OP_LOGI("Get ValueDepend Input StorageFormat %d, dataType %d, shape [%lu], addr is %p.",
            rt2Tensor->GetStorageFormat(), rt2Tensor->GetDataType(), array->Size(), rt2Tensor->GetAddr());
    return OK;
}

template <typename T>
static aclnnStatus NnopbaseExecutorAddArrayInput(NnopbaseTensors* tensors, const T* array, const uint32_t index)
{
    tensors->paramDescs.instances[index].isInput = true;
    if (array != nullptr) {
        const uint32_t startIndex = tensors->paramDescs.instances[index].startIndex;
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSaveArray<T>(array, &tensors->extTensors[startIndex]));
        tensors->paramDescs.instances[index].num = 1U;
        tensors->extTensors[startIndex].valueDepend = true;
        tensors->hostInputNum++;
        tensors->hostInputSize += tensors->extTensors[startIndex].rt2Tensor.GetSize();
        OP_LOGI("Executor add valuedepend input[%u] successfully.", index);
    } else {
        tensors->paramDescs.instances[index].num = 0U; // for null option
        tensors->paramDescs.emptyNum++;
    }
    return OK;
}
} // namespace nnopbase

#ifdef __cplusplus
extern "C" {
#endif
enum NnopbaseFormatCheckOption { kNnopbaseDefault = 0, kNnopbaseStrict, kNnopbaseMax };

enum class NnopbaseParamCheckMode {
    kCheckRequiredIo = 0,     // 默认只校验必选输入输出
    kCheckOptionalInput = 1,  // 默认校验必选输入输出+可选输入
    kCheckOptionalOutput = 2, // 默认校验必选输入输出+可选输出
    kCheckAllIo = 3,          // 默认校验所有输入输出
    kCheckEnd,
};

static inline bool IsContiguous(const GertShape& shape, const op::Strides& strides)
{
    OP_LOGI("Input tensor view shape is %s, view strides is %s", op::ToString(shape).GetString(),
            op::ToString(strides).GetString());
    int64_t valid_stride = 1;
    for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            continue;
        }
        if (valid_stride != strides[i]) {
            OP_LOGI("Input tensor is not contiguous, valid_stride is %ld, strides[%ld] is %ld", valid_stride, i,
                    strides[i]);
            return false;
        }
        valid_stride *= shape[i];
    }
    return true;
}

static inline bool IsContiguousShape(const aclTensor* tensor)
{
    const auto& view_shape = tensor->GetViewShape();
    if (view_shape.GetShapeSize() == 0 || view_shape.GetShapeSize() == 1) {
        return true;
    }
    return IsContiguous(view_shape, tensor->GetViewStrides());
}

inline void NnopbaseCheckContiguous(NnopbaseTensors* tensors, const aclTensor* tensor, const size_t index,
                                    const size_t irIdx)
{
    if (!IsContiguousShape(tensor)) {
        tensors->unContiguousTensors.tensors.push_back(tensor);
        tensors->unContiguousTensors.idxs.push_back(index);
        if (tensors->paramDescs.instances[irIdx].refIdx != -1) {
            tensors->unContiguousTensors.refUnContTensors.push_back(tensor);
            tensors->unContiguousTensors.refIdxs.push_back(index);
            // 构造连续的aclTensor，scalar一定不会进入非连续判断的分支，此处dim的数量一定不是0
            const int64_t* dims = &tensor->GetViewShape()[0U];
            const uint64_t dumNum = tensor->GetViewShape().GetDimNum();
            const auto contTensor = aclCreateTensor(dims, dumNum, op::ToAclDataType(tensor->GetDataType()), nullptr, 0,
                                                    op::ToAclFormat(tensor->GetViewFormat()), dims, dumNum,
                                                    tensor->GetData());
            tensors->unContiguousTensors.refContTensors.push_back(contTensor);
        }
        OP_LOGD("Op input irIdx [%zu] idx [%zu] refIdx[%d] is uncontiguous.", irIdx, index,
                tensors->paramDescs.instances[irIdx].refIdx);
    }
}

static inline void NnopbaseExecutorGet8ByteSize(size_t totalSize, uint32_t* len)
{
    if (totalSize % sizeof(uint64_t) == 0) {
        *len += totalSize;
    } else {
        *len += (totalSize / sizeof(uint64_t) + 1) * sizeof(uint64_t);
    }
}

static inline NnopbaseUChar* NnopbaseAppendBinary(void* buf, const size_t bufLen, const void* src, const size_t srcLen)
{
    if (src != nullptr && memcpy_s(buf, bufLen, src, srcLen) == EOK) {
        buf = op::internal::PtrCastTo<NnopbaseUChar>(buf) + srcLen;
    } else {
        OP_LOGW("Failed to append binary, probably because src is null or failed to execute memcpy_s!");
    }
    return op::internal::PtrCastTo<NnopbaseUChar>(buf);
}

static inline uint32_t NnopbaseExecutorGetTaskType(const CoreType coreType, NnopbaseTaskRation taskRation)
{
    switch (taskRation) {
        case kRation10:
            return MSPROF_GE_TASK_TYPE_MIX_AIC;
        case kRation01:
            return MSPROF_GE_TASK_TYPE_MIX_AIV;
        default:
            break;
    }

    switch (coreType) {
        case kMix:
            return MSPROF_GE_TASK_TYPE_MIX_AIC;
        case kVectorcore:
            return MSPROF_GE_TASK_TYPE_AIV;
        default:
            break;
    }
    return MSPROF_GE_TASK_TYPE_AI_CORE;
}

#ifdef __cplusplus
}
#endif

#endif
